# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
import functools


def check_kvc_concat(self):
    if 'unify_scales_for_kvc_concat' not in self.attrs:
        return False
    if not self.attrs['unify_scales_for_kvc_concat']:
        return False
    if len(self.parents) == 0:
        return False
    input_op = set()
    for inp in self.parents:
        input_op.add(inp.type)
    if len(input_op) == 2 and OpType.Input in input_op:
        return True
    return False


@quant_register(OpType.Concat)
def concat_quantize(self, *args):
    if check_kvc_concat(self):
        real_inp = None
        for inp in self.inputs:
            if inp.pnode.type != OpType.Input:
                real_inp = inp
                break
        for inp in self.inputs:
            if inp.pnode.type != OpType.Input:
                continue
            inp.clone_qinfo(real_inp)
        self.outputs[0].clone_qinfo(real_inp)
        return

    q_mode_activation = self.attrs["q_mode_activation"]
    q_bits_activation = self.attrs["q_bits_activation"]
    multiplier_bits = self.attrs['multiplier_bits']
    axis = self.get_param('axis')
    # #decide the output tensors' key properties related to quantization
    out = self.outputs[0]
    onumel = out.ir_shape.size()
    sign_branches = 0
    majority = 0
    majority_index = 0
    weighted_min = 0
    weighted_max = 0
    zps = []
    scs = []
    out.qinvariant = False
    qinv_s = []
    qin_bits = []
    for i, inp in enumerate(self.inputs):
        if is_signed(inp.dtype):
            sign_branches += 1
        bnumel = inp.ir_shape.size()
        if bnumel > majority:
            majority = bnumel
            majority_index = i
        alpha = bnumel / onumel
        weighted_min = weighted_min + alpha * inp.min
        weighted_max = weighted_max + alpha * inp.max
        zps.append(inp.zerop)
        scs.append(inp.scale)
        qinv_s.append(inp.qinvariant)
        qin_bits.append(inp.qbits)
        if inp.qinvariant:
            out.qinvariant = True
    if max(qinv_s) != min(qinv_s):
        OPT_WARN(f'{self}: some inputs is quantize invariant and other inputs is not, which may cause accuracy issue.',
                 workflow_name='quantize')

    sign = sign_branches > 0  # is_signed(self.inputs[min_index].dtype)
    # out.min = min(weighted_min, 0.0)
    # out.max = max(weighted_max, 0.0)
    out.qbits = max(qin_bits)  # q_bits_activation
    out.dtype = bits2dtype(out.qbits, is_signed=sign)

    """
    when enable activation perchannel, the concat inputs would have perchannel and pertensor input scales, so
    1. extend the input pertensor scale to perchannel scale
    2. use the perchannel scale to calculate the align scale or not.
    """

    def _broadcast_scale_zp(inputs, param_axis, output):
        is_multi_data = [is_torch_tensor_with_multi_data(ins.scale) for ins in inputs]
        key_axes = [ins.key_axis for ins in inputs]
        scales = [ins.scale for ins in inputs]
        zerops = [ins.zerop for ins in inputs]
        all_mutli_data = all(is_multi_data)
        any_multi_data = any(is_multi_data)
        out_key_axis = output.key_axis
        if all_mutli_data:
            return scales, zerops
        elif any_multi_data:
            idx = is_multi_data.index(True)
            is_same_axis = key_axes[idx] == param_axis
            act_axis = param_axis if is_same_axis else key_axes[idx]
            for i, is_perchannel_data in enumerate(is_multi_data):
                if not is_perchannel_data:
                    scales[i] = torch.full([inputs[i].ir_shape[act_axis]], scales[i].item(), device=inputs[i].device)
                    zerops[i] = torch.full([inputs[i].ir_shape[act_axis]], zerops[i].item(), device=inputs[i].device)
        elif out_key_axis is not None:
            for i, is_perchannel_data in enumerate(is_multi_data):
                if not is_perchannel_data:
                    scales[i] = torch.full([inputs[i].ir_shape[out_key_axis]],
                                           scales[i].item(), device=inputs[i].device)
                    zerops[i] = torch.full([inputs[i].ir_shape[out_key_axis]],
                                           zerops[i].item(), device=inputs[i].device)
        return scales, zerops

    scs, zps = _broadcast_scale_zp(self.inputs, axis, self.outputs[0])

    def _is_equal(scale_or_zp):
        cmp_0 = scale_or_zp[0]
        ret = []
        for i in range(1, len(scale_or_zp)):
            ret.append(torch.equal(cmp_0, scale_or_zp[i]))
        return ret

    flag = all(_is_equal(zps)) and all(_is_equal(scs)) and (sign_branches == 0 or sign_branches == len(self.inputs))
    if out.qinvariant:
        out.scale = 1.
        out.zerop = 0
        out.qbits = max(qin_bits)
        out.dtype = bits2dtype(out.qbits, is_signed=sign)
    elif flag:
        out.scale = self.inputs[0].scale
        out.zerop = self.inputs[0].zerop
        out.qmin, out.qmax = bits2range(out.qbits, sign)
    else:
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, out.qbits, sign)
    zps.append(out.zerop)
    scs.append(out.scale)
    branch_scales = []
    branch_types1 = []
    branch_shifts = []
    branch_types2 = []

    o_scale = []
    o_zerop = []
    out_key_axis = self.outputs[0].key_axis
    same_with_axis = out_key_axis == axis
    if out.is_perchannel_quantization() and same_with_axis:
        for i, inp in enumerate(self.inputs):
            o_scale.append(scs[i])
            o_zerop.append(zps[i])
        out.scale = torch.cat(o_scale, dim=0).to(self.inputs[0].device)
        out.zerop = torch.cat(o_zerop, dim=0).to(self.inputs[0].device)
        self.attrs['need_align_scales'] = False
    else:
        bs_threshold = float(self.get_attrs('unify_scales_for_multi_inputs_operator_threshold',
                                            optional=True, default_value=1.0))
        max_sc, min_sc = max([sc.max().item() for sc in scs]), min([sc.min().item() for sc in scs])
        max_zp, min_zp = max([zp.max().item() for zp in zps]), min([zp.min().item() for zp in zps])
        if ((max_sc / (min_sc + OPT_EPSILON)) <= bs_threshold and (max_zp - min_zp) < OPT_EPSILON) or out.qinvariant:
            self.attrs['need_align_scales'] = False
            OPT_DEBUG("layer_id=%s, %s, %s : this concat does not need to align scale/zerop" %
                      (self.attrs['layer_id'], str(self.type), self.name))
            self.attrs['optimization_info']['unify_scales_for_multi_inputs_operator'] = True
        else:
            self.attrs['need_align_scales'] = True

    if self.attrs['need_align_scales']:
        o_scale = out.scale
        for i, inp in enumerate(self.inputs):
            s = o_scale / inp.scale
            multipiler, multipiler_type, shift, shift_type = \
                get_scale_approximation_params(s, multiplier_bits, force_shift_positive=self.force_shift_positive)

            branch_scales.append(multipiler.int())
            branch_shifts.append(shift.int())
            branch_types1.append(multipiler_type)
            branch_types2.append(shift_type)

        scale_name = 'scale' if is_torch_tensor_with_multi_data(branch_scales[0]) else 'scale_value'
        shift_name = 'shift' if is_torch_tensor_with_multi_data(branch_shifts[0]) else 'shift_value'
        '''
        why doscales use torch.stack, not torch.cat in activation perchannel, which may have different
        branch_scales shape and will cause code crash, because if activation_perchannel_concat_in_axis == true,
        need_align_scales=false, donot generate the scale/shift field. if need_align_scales==true, the all branch
        doscale are the same shape, so we can stack, this is convenience for forward to get the branch scale.
        '''
        do_scales = torch.stack(branch_scales, dim=0) if is_torch_tensor_with_multi_data(
            branch_scales[0]) else [bs.item() if is_torch_tensor(bs) else bs for bs in branch_scales]
        do_shifts = torch.stack(branch_shifts, dim=0) if is_torch_tensor_with_multi_data(
            branch_shifts[0]) else [bs.item() if is_torch_tensor(bs) else bs for bs in branch_shifts]
        self.set_ir_field(scale_name, do_scales, branch_types1[0])
        self.set_ir_field(shift_name, do_shifts, branch_types2[0])
        if not is_torch_tensor_with_multi_data(branch_scales[0]):
            self.params["scale_type"] = branch_types1
            self.params["shift_type"] = branch_types2


@op_register(OpType.Concat)
def concat(self, *args):
    axis = self.get_param('axis')
    out = self.outputs[0]
    inp_betensors = []
    if check_kvc_concat(self):
        real_inp = None
        for inp in self.inputs:
            if inp.pnode.type != OpType.Input:
                real_inp = inp
                break
        for inp in self.inputs:
            if inp.pnode.type != OpType.Input:
                continue
            inp.betensor = real_inp.betensor.clone()
    for i, inp in enumerate(self.inputs):
        inp_betensors.append(inp.betensor)
    if self.quantized:
        in_t = []
        branch_scales = self.get_ir_field(['scale_value', 'scale'], default_value=[1] * len(self.inputs))
        branch_shifts = self.get_ir_field(['shift_value', 'shift'], default_value=[0] * len(self.inputs))
        same_with_axis = self.outputs[0].key_axis == axis
        for i, inp in enumerate(self.inputs):
            doscale = branch_scales[i]
            doshift = branch_shifts[i]
            zerop = out.broadcast_zerop
            inp_zerop = inp.broadcast_zerop
            if is_torch_tensor_with_multi_data(doscale):
                doscale = branch_scales[i].reshape(inp.key_axis_broadcast_shape())
                doshift = branch_shifts[i].reshape(inp.key_axis_broadcast_shape())
            if is_torch_tensor_with_multi_data(zerop) and same_with_axis:
                axis_shape = [inp.ir_shape[inp.key_axis] if inp.key_axis is not None else 1 for inp in self.inputs]
                start = functools.reduce(lambda x, y: x + y, axis_shape[:i]) if i > 0 else 0
                zerop = out.zerop[start: start + axis_shape[i]].reshape(inp.key_axis_broadcast_shape())
            t = linear_requantize(inp.betensor + inp_zerop, doscale, doshift, zerop, out.qmin, out.qmax)
            in_t.append(t)
        out.betensor = torch.cat(in_t, dim=axis)  # NHWC
    else:
        out.betensor = torch.cat(inp_betensors, dim=axis)  # NHWC
    return out.betensor
