# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
from AIPUBuilder.Optimizer.ops.activation import apply_with_activation, with_activation_out_is_signed, \
    apply_with_activation_quantize
import torch


def calc_eltwise_add_like_scale_shift(inp0, inp1, out, doscale_clip_max, multiplier_bits, layer_type,
                                      layer_id='unknow'):
    # ######################################################################################################
    # # former schema
    # clip_max = doscale_clip_max
    # g_eltwise_scale_bits = 8
    # inp_scale_max = max(inp0.scale, inp1.scale)
    # # avoid to warning occurrence, we need ignore the relative extreme big/small scale, so
    # # proof_min_ration is defined
    # proof_min_ration = (2**g_eltwise_scale_bits)/clip_max
    # inp0_scale = inp0.scale
    # inp1_scale = inp1.scale
    # # it had better to avoid entering proof_min_ration, so try to reduce g_eltwise_scale_bits
    # # example, inp0.scale=19804, inpi.scale=13182574, if g_eltwise_scale_bits is 8, inp_scale_max will be changed, and affect acc
    # # but if g_eltwise_scale_bits reduce to 5, inp_scale_max keep unchanged
    # while (inp0.scale/inp1.scale < proof_min_ration or inp1.scale/inp0.scale < proof_min_ration) and g_eltwise_scale_bits > 1:
    #     g_eltwise_scale_bits = g_eltwise_scale_bits-1
    #     proof_min_ration = (2**g_eltwise_scale_bits)/clip_max
    # if g_eltwise_scale_bits == 1:
    #     g_eltwise_scale_bits = 8
    #     # proof_min_ration = (2**g_eltwise_scale_bits)/clip_max
    #     # if inp0.scale/inp1.scale < proof_min_ration or inp1.scale/inp0.scale < proof_min_ration:
    #     inp0_scale = min(max(inp0.scale, 1./clip_max), clip_max)
    #     inp1_scale = min(max(inp1.scale, 1./clip_max), clip_max)
    #     inp_scale_max = max(inp0_scale, inp1_scale)

    # if inp0.qinvariant and not inp1.qinvariant:
    #     inp_scale_max = inp0.scale
    # if inp1.qinvariant and not inp0.qinvariant:
    #     inp_scale_max = inp1.scale

    # scale0 = (inp_scale_max / inp0_scale) * (2**g_eltwise_scale_bits)
    # scale1 = (inp_scale_max / inp1_scale) * (2**g_eltwise_scale_bits)

    # while int(scale0) > clip_max or int(scale1) > clip_max:
    #     if scale0 == 1:  # scale1>clip_max, but scale0=1,
    #         (OPT_DEBUG(f"layer_id={layer_id}, layer_type={layer_type}, the second scale={int(scale1)} of inputs "
    #                    f"has out range [0, {int(clip_max)}], please attention."))
    #     if scale1 == 1:
    #         (OPT_DEBUG(f"layer_id={layer_id}, layer_type={layer_type}, the second scale={int(scale0)} of inputs "
    #                    f"has out range [0, {int(clip_max)}], please attention."))
    #     scale0 = max(round(scale0 / 2), 1)  # to avoid one scale to be 0
    #     scale1 = max(round(scale1 / 2), 1)  # to avoid one scale to be 0
    #     g_eltwise_scale_bits -= 1

    # local_rescale = out.scale / (inp_scale_max)
    # do_scale, do_scale_type, do_shift, do_shift_type = \
    #     get_scale_approximation_params(local_rescale / (2**g_eltwise_scale_bits),
    #                                     mult_bits=multiplier_bits,
    #                                     force_shift_positive=True)
    # plh_scale = max(inp0.scale, inp1.scale) * (2**g_eltwise_scale_bits)
    # return scale0, scale1, do_scale, do_shift, do_scale_type, do_shift_type, plh_scale
    ######################################################################################################
    # experimental schema
    # Yq = ((Aq+ZPa)Sb +(Bq+ZPb)Sa)Sy/SaSb -ZPy
    # replace Sa, Sb, Sy/SaSb with M0/2^N0, M1/2^N1, M2/2^N2,
    # Yq = ((Aq+ZPa)M_1*2^(N_0-N_1) +(Bq+ZPb)M_0)M_2/2^(N_0+N_2) -ZPy
    # assueme N_0 >= N_1
    cbits = range2dtype(0, doscale_clip_max)[0]
    doscale_clip_max = torch_tensor(doscale_clip_max, device=inp0.device)
    M0, _, N0, _ = get_scale_approximation_params(inp0.scale, mult_bits=cbits)
    M1, _, N1, _ = get_scale_approximation_params(inp1.scale, mult_bits=cbits)
    cshift = torch.zeros_like(N0)
    scale0 = torch.ones_like(M0)
    scale1 = torch.ones_like(scale0)
    cshift = torch.where(N0 > N1, N0, N1)
    diff0 = torch.max((cshift - N0), torch.zeros_like(cshift))
    diff1 = torch.max((cshift - N1), torch.zeros_like(cshift))
    scale0 = M1 * 2 ** diff1
    scale1 = M0 * 2 ** diff0
    rscale = out.scale / (inp1.scale * inp0.scale)
    rscale = rscale / (2.0 ** cshift)
    max_shrink = torch.max(torch.max(scale0, scale1)) * 1.0 / doscale_clip_max
    if max_shrink > 1.0:
        if torch.min(torch.min(scale0, scale1)) / max_shrink >= 1.0:
            scale0 = scale0 / max_shrink
            scale1 = scale1 / max_shrink
            rscale = rscale * max_shrink
        else:
            # shrink = min(scale0, scale1)
            scale0 = scale0 / max_shrink
            scale1 = scale1 / max_shrink
            rscale = rscale * max_shrink
            OPT_DEBUG(f"layer_id={layer_id}, layer_type={layer_type}, the input scales={(inp0.scale, inp1.scale)} "
                      f"are very disproportional and caused out of range scale value during quantization, please pay attention.")
    one_t = torch_tensor(1, device=inp0.device)
    zero_t = torch_tensor(0, device=inp0.device)
    scale0 = torch.max(zero_t, torch.min(doscale_clip_max, torch.round(scale0)))
    scale1 = torch.max(zero_t, torch.min(doscale_clip_max, torch.round(scale1)))
    do_scale, _, do_shift, _ = get_scale_approximation_params(rscale, mult_bits=multiplier_bits)
    shift_less0_mask = do_shift < 0
    do_scale[shift_less0_mask] = torch.max(one_t, torch.min(doscale_clip_max,
                                                            do_scale[shift_less0_mask] * (
                                                                torch.pow(2, (do_shift[shift_less0_mask]).abs()))))
    do_shift[shift_less0_mask] = 0

    plh_scale = inp1.scale * inp0.scale
    return (scale0, scale1, do_scale, do_shift, range2dtype(0, do_scale.max().item())[1],
            range2dtype(-1, do_shift.max().item())[1], plh_scale)
    ######################################################################################################


def eltwise_quantizes(self, *args):
    inp0, inp1 = self.inputs[0], self.inputs[1]
    out = self.outputs[0]
    method = self.get_param("method").upper()
    q_mode_activation = self.attrs["q_mode_activation"]
    multiplier_bits = self.attrs['multiplier_bits']
    q_bits_activation = self.attrs["q_bits_activation"]
    act_type = self.get_param('with_activation', optional=True, default_value='none').lower()
    if act_type == 'none':
        out_signed = is_signed(inp0.dtype) or is_signed(inp1.dtype) or (method == 'SUB')
    else:
        out_signed = with_activation_out_is_signed(self) or self.force_dtype_int
    if inp0.qinvariant != inp1.qinvariant:
        OPT_WARN(f"{self} one input is quantize invariant and other one input is not, which may cause accuracy issue.",
                 workflow_name='quantize')

    if inp0.qinvariant and inp1.qinvariant:
        out.scale = 1.0
        out.zerop = 0
        out.qbits, _ = range2dtype(out.extrema_min, out.extrema_max, force_int=out_signed)
        out.qbits = max(out.qbits, max(inp0.qbits, inp1.qbits))
        out.dtype = bits2dtype(out.qbits, is_signed=out_signed)
        out.qinvariant = True
    else:
        out.qinvariant = False
        out.qbits = q_bits_activation
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, out.qbits, is_signed=out_signed)

    if method in {"ADD", "SUB", "MAX", "MIN", "NONE"}:
        # due to aiff don't support uint16 max 65535,so we use INT16 replace UINT16
        _, clip_max = dtype2range(Dtype.INT16) if 'clip_max_bits' not in self.attrs else bits2range(
            self.attrs['clip_max_bits'], True)
        scale0, scale1, do_scale, do_shift, do_scale_type, do_shift_type, _ = calc_eltwise_add_like_scale_shift(
            inp0, inp1, out, torch_tensor(clip_max, inp0.device), multiplier_bits, self.type, self.attrs["layer_id"])

        bs_threshold = float(self.get_attrs('unify_scales_for_multi_inputs_operator_threshold',
                                            optional=True, default_value=1.0))
        if all((torch.maximum(inp0.scale, inp1.scale) / (
                torch.minimum(inp0.scale, inp1.scale) + OPT_EPSILON)) <= bs_threshold) and all(
                (inp0.zerop - inp1.zerop).abs() <= OPT_EPSILON):
            if method in {"ADD", }:
                do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
                    out.scale / inp0.scale, mult_bits=multiplier_bits, force_shift_positive=self.force_shift_positive)
            else:
                do_scale = torch_tensor(1, device=inp0.device)
                do_shift = torch_tensor(0, device=inp0.device)
            scale0 = torch_tensor(1, device=inp0.device)
            scale1 = torch_tensor(1, device=inp0.device)
            self.attrs['need_align_scales'] = False
            OPT_DEBUG(f"{self} this layer does not need to align input branches' scale/zerop.")
            self.attrs['optimization_info']['unify_scales_for_multi_inputs_operator'] = True

        scale_name = 'scale' if is_torch_tensor_with_multi_data(scale0) else 'scale_value'
        shift_name = 'shift' if is_torch_tensor_with_multi_data(do_shift) else 'shift_value'
        key_axis_shape = out.ir_shape[out.key_axis] if out.key_axis is not None else 1
        if key_axis_shape > 1:
            scale0 = torch.full([key_axis_shape], scale0.item()).to(
                inp0.device) if not is_torch_tensor_with_multi_data(scale0) else scale0
            scale1 = torch.full([key_axis_shape], scale1.item()).to(
                inp1.device) if not is_torch_tensor_with_multi_data(scale1) else scale1
            do_scale = torch.full([key_axis_shape], do_scale.item()).to(
                out.device) if not is_torch_tensor_with_multi_data(do_scale) else do_scale

        do_scales = torch.stack([do_scale.to(dtype2torch_type(Dtype.UINT16)), scale0, scale1], dim=0) \
            if is_torch_tensor_with_multi_data(scale0) else [do_scale.int().item(), scale0.int().item(), scale1.int().item()]
        self.set_ir_field(scale_name, do_scales, Dtype.UINT16)
        self.set_ir_field(shift_name, do_shift, do_shift_type)
        if not is_torch_tensor_with_multi_data(scale0):
            self.params["scale_type"] = [Dtype.UINT16, Dtype.UINT16, Dtype.UINT16]
            self.params["shift_type"] = do_shift_type

    elif method == "MUL":
        local_rescale = out.scale / (inp0.scale * inp1.scale)
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(local_rescale, mult_bits=multiplier_bits,
                                           force_shift_positive=self.force_shift_positive)
        doscale_name = 'scale' if is_torch_tensor_with_multi_data(do_scale) else 'scale_value'
        doshift_name = 'shift' if is_torch_tensor_with_multi_data(do_shift) else 'shift_value'
        self.set_ir_field(doscale_name, do_scale, do_scale_type)
        self.set_ir_field(doshift_name, do_shift, do_shift_type)
        if not is_torch_tensor_with_multi_data(do_scale):
            self.params["shift_type"] = do_shift_type
            self.params["scale_type"] = do_scale_type


@op_register(OpType.Eltwise)
def eltwise(self, *args):
    '''
    eltwise op
    eltwise op has five kind of method:
    * ADD -> add
    * SUB -> sub
    * MUL -> mul
    * MAX -> max
    * MIN -> min
    each kind of method can append any type of relu(activation)
    '''
    method = self.get_param("method").upper()
    if method not in {"ADD", "SUB", "MUL", "MAX", "MIN"}:
        OPT_FATAL("unsupported method: %s for eltwise in node:%s" % (method, self.name))

    method_d = {
        "ADD": lambda a, b: a + b,
        "SUB": lambda a, b: a - b,
        "MUL": lambda a, b: a * b,
        "MAX": lambda a, b: torch.max(a, b),
        "MIN": lambda a, b: torch.min(a, b),
    }
    op = method_d[method]
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]

    def set_can_detile(t):
        if t.pnode is None:
            can_detile = False
        else:
            pow_nods, count_root, count_constant = t.pnode.get_ancestors()
            can_detile = True if count_root > 0 and count_root == count_constant else False
        return can_detile

    inp0_can_detile = self.get_attrs('inp0_can_detile', optional=True, default_value=None)
    inp1_can_detile = self.get_attrs('inp1_can_detile', optional=True, default_value=None)
    if inp0_can_detile is None:
        inp0_can_detile = set_can_detile(inp0)
        self.attrs['inp0_can_detile'] = inp0_can_detile
    if inp1_can_detile is None:
        inp1_can_detile = set_can_detile(inp1)
        self.attrs['inp1_can_detile'] = inp1_can_detile
    x0 = inp0.betensor
    x1 = inp1.betensor
    if inp0_can_detile:
        x0 = inp0.detile_betensor()
    if inp1_can_detile:
        x1 = inp1.detile_betensor()
    x0 = x0.to(torch.int64) if self.quantized else x0.float()
    x1 = x1.to(torch.int64) if self.quantized else x1.float()

    if method in {"ADD", "SUB", "MAX", "MIN"}:
        if self.quantized:
            scales = self.get_ir_field(['scale_value', 'scale'])
            scale0, scale1 = scales[1], scales[2]
            # deduce ensure out.key_axis is the same with inp0.key_axis or inp1.key_axis when inp0.key_axis/inp1.key_axis is not None
            x0 = linear_requantize(x0 + inp0.broadcast_zerop, scale0, 0, 0, -2 ** 31, 2 ** 31, key_axis=out.key_axis)
            x1 = linear_requantize(x1 + inp1.broadcast_zerop, scale1, 0, 0, -2 ** 31, 2 ** 31, key_axis=out.key_axis)
    elif method in {"MUL"}:
        if self.quantized:
            x0 = x0 + inp0.broadcast_zerop
            x1 = x1 + inp1.broadcast_zerop
    x0shape = list(x0.shape)
    x1shape = list(x1.shape)
    x0dims = len(x0shape)
    x1dims = len(x1shape)
    # broadcasting, shape align
    x0, x1 = broadcasting_transform(x0, x1)
    x = op(x0, x1)

    bk_scale = None
    if self.quantized:
        bk_scale = self.get_ir_field(['scale_value', 'scale'])
        if method in {"ADD", "SUB", "MAX", "MIN"}:
            scale_name = 'scale' if is_torch_tensor_with_multi_data(bk_scale[0]) else 'scale_value'
            self.set_ir_field(scale_name, bk_scale[0])
    out.betensor = apply_with_activation(self, x, *args)
    if bk_scale is not None and method in {"ADD", "SUB", "MAX", "MIN"}:
        scale_name = 'scale' if is_torch_tensor_with_multi_data(bk_scale[0]) else 'scale_value'
        self.set_ir_field(scale_name, bk_scale)
    return out.betensor


@quant_register(OpType.Eltwise)
def eltwise_quantize(self, *args):
    eltwise_quantizes(self, *args)
    apply_with_activation_quantize(self, self.outputs[0].qinvariant, *args)
