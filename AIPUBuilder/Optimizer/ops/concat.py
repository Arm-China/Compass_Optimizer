# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *


@quant_register(OpType.Concat)
def concat_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Concat currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]
    multiplier_bits = self.attrs['multiplier_bits']
    # #decide the output tensors' key properties related to quantization
    out = self.outputs[0]
    dev = out.device
    onumel = 1
    for s in out.ir_shape:
        onumel *= s
    min_scale = torch_tensor(float("inf"), device=dev)
    min_index = torch_tensor(0, device=dev)
    max_scale = torch_tensor(float("-inf"), device=dev)
    max_index = torch_tensor(0, device=dev)
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
        if inp.scale < min_scale:
            min_scale = inp.scale
            min_index = i
        if inp.scale > max_scale:
            max_scale = inp.scale
            max_index = i
        if is_signed(inp.dtype):
            sign_branches += 1
        bnumel = 1
        for s in inp.ir_shape:
            bnumel *= s
        if bnumel > majority:
            majority = bnumel
            majority_index = i
        alpha = bnumel / onumel
        weighted_min = weighted_min + alpha * inp.min
        weighted_max = weighted_max + alpha * inp.max
        # currently not support per-channel activation quantization,so inp.scale and inp.zerop numel is 1
        if isinstance(inp.scale, torch.Tensor) and inp.scale.numel() == 1:
            zps.append(inp.zerop.item())
            scs.append(inp.scale.item())
        else:
            zps.append(inp.zerop)
            scs.append(inp.scale)
        qinv_s.append(inp.qinvariant)
        qin_bits.append(inp.qbits)
        if inp.qinvariant:
            out.qinvariant = True
    if max(qinv_s) != min(qinv_s):
        OPT_WARN('some inputs is quantize invariant and other inputs is not, which may cause accuracy issue. layer_id=%s, %s' % (self.attrs['layer_id'], self.name),
                 workflow_name='quantize', op_name=str(self.type))

    sign = sign_branches > 0  # is_signed(self.inputs[min_index].dtype)
    # out.min = min(weighted_min, 0.0)
    # out.max = max(weighted_max, 0.0)
    out.qbits = max(qin_bits)  # q_bits_activation
    out.dtype = bits2dtype(out.qbits, is_signed=sign)
    if out.qinvariant:
        out.scale = 1.
        out.zerop = 0
        out.qbits = max(qin_bits)
        out.dtype = bits2dtype(out.qbits, is_signed=sign)
    elif (len(set(zps)) == 1) and (len(set(scs)) == 1) and (sign_branches == 0 or sign_branches == len(self.inputs)):
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
    for i, inp in enumerate(self.inputs):
        s = out.scale / inp.scale
        multipiler, multipiler_type, shift, shift_type = get_scale_approximation_params(s, multiplier_bits,
                                                                                        force_shift_positive=self.force_shift_positive)
        branch_scales.append(int(multipiler))
        branch_shifts.append(int(shift))
        branch_types1.append(multipiler_type)
        branch_types2.append(shift_type)

    bs_threshold = float(self.get_attrs('unify_scales_for_multi_inputs_operator_threshold',
                                        optional=True, default_value=1.0))
    if ((max(scs) / (min(scs)+OPT_EPSILON)) <= bs_threshold and (max(zps) - min(zps)) < OPT_EPSILON) or out.qinvariant:
        self.attrs['need_align_scales'] = False
        OPT_DEBUG("layer_id=%s, %s, %s : this concat does not need to align scale/zerop" %
                  (self.attrs['layer_id'], str(self.type), self.name))
        self.attrs['optimization_info']['unify_scales_for_multi_inputs_operator'] = True
    else:
        self.attrs['need_align_scales'] = True

    if self.attrs['need_align_scales']:
        self.params['scale_type'] = branch_types1
        self.params['scale_value'] = branch_scales
        self.params['shift_type'] = branch_types2
        self.params['shift_value'] = branch_shifts


@op_register(OpType.Concat)
def concat(self, *args):
    axis = self.get_param('axis')
    out = self.outputs[0]
    inp_betensors = []
    for i, inp in enumerate(self.inputs):
        inp_betensors.append(inp.betensor)
    if self.quantized and 'scale_value' in self.params:
        in_t = []
        branch_scales = self.params['scale_value']
        branch_shifts = self.params['shift_value']
        for i, inp in enumerate(self.inputs):
            t = linear_requantize(inp.betensor + inp.zerop,
                                  branch_scales[i], branch_shifts[i], out.zerop, out.qmin, out.qmax)
            in_t.append(t)
        out.betensor = torch.cat(in_t, dim=axis)  # NHWC
    else:
        out.betensor = torch.cat(inp_betensors, dim=axis)  # NHWC
    return out.betensor
