# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.ops.activation import apply_with_activation, with_activation_out_is_signed, apply_with_activation_quantize
import torch


def calc_eltwise_add_like_scale_shift(inp0, inp1, out, doscale_clip_max, layer_type, layer_id='unknow'):
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
    #                                     mult_bits=out.qbits,
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
    M0, _, N0, _ = get_scale_approximation_params(inp0.scale, mult_bits=cbits)
    M1, _, N1, _ = get_scale_approximation_params(inp1.scale, mult_bits=cbits)
    cshift = 0
    scale0 = 1
    scale1 = 1
    if N0 > N1:
        cshift = N0
        scale0 = M1 * (2.0 ** (N0-N1))
        scale1 = M0
    else:
        cshift = N1
        scale0 = M1
        scale1 = M0 * (2.0 ** (N1-N0))
    rscale = out.scale / (inp1.scale * inp0.scale)
    rscale = rscale / (2.0**cshift)
    max_shrink = max(scale0, scale1) * 1.0 / doscale_clip_max
    if max_shrink > 1.0:
        if min(scale0, scale1) / max_shrink >= 1.0:
            scale0 /= max_shrink
            scale1 /= max_shrink
            rscale *= max_shrink
        else:
            # shrink = min(scale0, scale1)
            scale0 /= max_shrink
            scale1 /= max_shrink
            rscale *= max_shrink
            OPT_DEBUG(f"layer_id={layer_id}, layer_type={layer_type}, the input scales={(inp0.scale, inp1.scale)} "
                      f"are very disproportional and caused out of range scale value during quantization, please pay attention.")
    scale0 = max(0, min(doscale_clip_max, round(scale0)))
    scale1 = max(0, min(doscale_clip_max, round(scale1)))
    do_scale, _, do_shift, _ = get_scale_approximation_params(rscale, mult_bits=out.qbits)
    if do_shift < 0:
        do_scale = max(1, min(doscale_clip_max, do_scale * (2.0 ** abs(do_shift))))
        do_shift = 0
    plh_scale = inp1.scale * inp0.scale
    return scale0, scale1, do_scale, do_shift, range2dtype(0, do_scale)[1], range2dtype(-1, do_shift)[1], plh_scale
    ######################################################################################################


def eltwise_quantizes(self, *args):
    inp0, inp1 = self.inputs[0], self.inputs[1]
    out = self.outputs[0]
    method = self.get_param("method").upper()
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Eltwise currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]
    act_type = self.get_param('with_activation', optional=True, default_value='none').lower()
    if act_type == 'none':
        out_signed = is_signed(inp0.dtype) or is_signed(inp1.dtype) or (method == 'SUB')
    else:
        out_signed = with_activation_out_is_signed(self) or self.force_dtype_int
    if inp0.qinvariant != inp1.qinvariant:
        OPT_WARN(
            'one input is quantize invariant and other one input is not, which may cause accuracy issue. layer_id=%s, %s' % (
                self.attrs['layer_id'], self.name),
            workflow_name='quantize', op_name=str(self.type))

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
        _, clip_max = dtype2range(Dtype.INT16)
        scale0, scale1, do_scale, do_shift, do_scale_type, do_shift_type, plh_scale = calc_eltwise_add_like_scale_shift(
            inp0, inp1, out, clip_max, self.type, self.attrs["layer_id"])
        # placeholders is only useful for ideal_mode (to help recording the float scale of var before activation)
        # which is now deprecated and can be remove in future
        if len(self.placeholders) < 1:
            tensor_name = self.graph.get_valid_tensor_name(self.name)
            ph0 = PyTensor(tensor_name, torch.tensor(0.).cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            self.placeholders.append(ph0)
        self.placeholders[0].scale = plh_scale

        bs_threshold = float(self.get_attrs('unify_scales_for_multi_inputs_operator_threshold',
                             optional=True, default_value=1.0))
        if (max(inp0.scale, inp1.scale) / (min(inp0.scale, inp1.scale) + OPT_EPSILON)) <= bs_threshold and abs(inp0.zerop - inp1.zerop) <= OPT_EPSILON:
            if method in {"ADD", }:
                do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
                    out.scale / inp0.scale, mult_bits=q_bits_activation, force_shift_positive=self.force_shift_positive)
            else:
                do_scale = 1
                do_shift = 0
            scale0 = 1
            scale1 = 1
            self.attrs['need_align_scales'] = False
            OPT_DEBUG("layer_id=%s, %s, %s : this layer does not need to align input branches' scale/zerop" %
                      (self.attrs['layer_id'], str(self.type), self.name))
            self.attrs['optimization_info']['unify_scales_for_multi_inputs_operator'] = True

        self.params["shift_value"] = int(do_shift)
        self.params["shift_type"] = do_shift_type
        self.params["scale_value"] = [int(do_scale), int(scale0), int(scale1)]
        self.params["scale_type"] = [do_scale_type, Dtype.UINT16, Dtype.UINT16]

    elif method == "MUL":
        local_rescale = out.scale / (inp0.scale * inp1.scale)
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(local_rescale, mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)
        self.params["shift_value"] = int(do_shift)
        self.params["shift_type"] = do_shift_type
        self.params["scale_value"] = int(do_scale)
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
    x0 = inp0.betensor.float()
    x1 = inp1.betensor.float()
    x_scale = 1.0
    x_zerop = 0
    if len(self.placeholders) < 1:
        tensor_name = self.graph.get_valid_tensor_name(self.name)
        ph0 = PyTensor(tensor_name, torch.tensor(0.).cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
        self.placeholders.append(ph0)
    if method in {"ADD", "SUB", "MAX", "MIN"}:
        if self.quantized:
            scale, scale0, scale1 = self.params["scale_value"]
            x0 = (x0 + inp0.zerop) * scale0
            x1 = (x1 + inp1.zerop) * scale1
        x_scale = self.placeholders[0].scale
    elif method in {"MUL"}:
        if self.quantized:
            x0 = x0 + inp0.zerop
            x1 = x1 + inp1.zerop
        x_scale = inp0.scale * inp1.scale
    x0shape = list(x0.shape)
    x1shape = list(x1.shape)
    x0dims = len(x0shape)
    x1dims = len(x1shape)
    # broadcasting
    if x0dims > x1dims and x1dims > 0:
        x1shape_ext = []
        dx_pre = -1
        for d in x1shape:
            dx = x0shape[dx_pre+1:].index(d) + dx_pre+1
            x1shape_ext.extend([1]*(dx-dx_pre))
            x1shape_ext[dx] = d
            dx_pre = dx
        x1shape_ext.extend([1]*(x0dims-len(x1shape_ext)))
        x1shape = x1shape_ext
    elif x1dims > x0dims and x0dims > 0:
        x0shape_ext = []
        dx_pre = -1
        for d in x0shape:
            dx = x1shape[dx_pre+1:].index(d) + dx_pre+1
            x0shape_ext.extend([1]*(dx-dx_pre))
            x0shape_ext[dx] = d
            dx_pre = dx
        x0shape_ext.extend([1]*(x1dims-len(x0shape_ext)))
        x0shape = x0shape_ext
    else:
        pass
    if len(x1shape) > x1dims:
        OPT_DEBUG(
            f'the second input broadcasting from {list(x1.shape)} to {x1shape} in node: {self.name}', log_once=True)
    if len(x0shape) > x0dims:
        OPT_DEBUG(
            f'the first input broadcasting from {list(x0.shape)} to {x0shape} in node: {self.name}', log_once=True)
    x = op(x0.reshape(x0shape), x1.reshape(x1shape))

    requant_scale = 1
    requant_shift = 0
    if self.quantized:
        if 'scale_value' in self.params:
            requant_scale = self.params['scale_value']
            if method in {"ADD", "SUB", "MAX", "MIN"}:
                requant_scale = requant_scale[0]
        elif "scale" in self.constants:
            requant_scale = self.constants["scale"].betensor
            if method in {"ADD", "SUB", "MAX", "MIN"}:
                requant_scale = requant_scale[0]

        if 'shift_value' in self.params:
            requant_shift = self.params['shift_value']
        elif "shift" in self.constants:
            requant_shift = self.constants["shift"].betensor
    # top_type_original = self.attrs['layer_top_type_original'][0]
    # original_top_dtype = str2dtype(top_type_original)
    # is_original_top_float = is_float(original_top_dtype)
    # if True == is_original_top_float :
    x = apply_with_activation(self, x,
                              x_scale, x_zerop,
                              requant_scale,
                              requant_shift,
                              *args)

    out.betensor = x
    return out.betensor


@quant_register(OpType.Eltwise)
def eltwise_quantize(self, *args):
    eltwise_quantizes(self, *args)
    apply_with_activation_quantize(self, self.outputs[0].qinvariant, *args)
