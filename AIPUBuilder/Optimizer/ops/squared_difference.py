# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch

g_eltwise_scale_bits = 8


@quant_register(OpType.SquaredDifference)
def squareddifference_quantizes(self, *args):
    inp0, inp1 = self.inputs[0], self.inputs[1]
    out = self.outputs[0]
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Eltwise currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    out_signed = False or self.force_dtype_int
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

    oscale = out.scale
    # SUB
    inp_scale_max = max(inp0.scale, inp1.scale)
    # due to aiff don't support uint16 max 65535,so we use INT16 replace UINT16
    _, clip_max = dtype2range(Dtype.INT16)
    # avoid to warning occurrence, we need ignore the relative extreme big/small scale, so
    # proof_min_ration is defined
    proof_min_ration = (2**g_eltwise_scale_bits)/clip_max
    inp0_scale = inp0.scale
    inp1_scale = inp1.scale
    if inp0.scale/inp1.scale < proof_min_ration or inp1.scale/inp0.scale < proof_min_ration:
        inp0_scale = min(max(inp0.scale, 1./clip_max), clip_max)
        inp1_scale = min(max(inp1.scale, 1./clip_max), clip_max)
        inp_scale_max = max(inp0_scale, inp1_scale)

    if inp0.qinvariant and not inp1.qinvariant:
        inp_scale_max = inp0.scale
    if inp1.qinvariant and not inp0.qinvariant:
        inp_scale_max = inp1.scale

    scale0 = (inp_scale_max / inp0_scale) * (2**g_eltwise_scale_bits)
    scale1 = (inp_scale_max / inp1_scale) * (2**g_eltwise_scale_bits)

    if int(scale0) > clip_max:
        (OPT_DEBUG('layer_id=%s, layer_type=%s the first scale=%d of eltwise has out range [0, 65535], please attention.'
                   % (self.attrs['layer_id'], str(self.type), int(scale0))))
        scale0 = min(scale0, clip_max)
    if int(scale1) > clip_max:
        (OPT_DEBUG('layer_id=%s, layer_type=%s the second scale=%d of eltwise has out range [0, 65535], please attention.'
                   % (self.attrs['layer_id'], str(self.type), int(scale1))))
        scale1 = min(scale1, clip_max)

    placeholders = self.placeholders[0]
    placeholders.qbits = 16
    placeholders.scale, placeholders.zerop, placeholders.qmin, placeholders.qmax, placeholders.dtype = get_linear_quant_params_from_tensor(
        placeholders, QuantMode.to_symmetric(q_mode_activation), placeholders.qbits, is_signed=True)
    placeholders.qinvariant = False
    scale_minus = placeholders.scale

    local_rescale = oscale / (scale_minus*scale_minus)
    do_scale_minus, do_scale_minus_type, do_shift_minus, do_shift_minus_type = \
        get_scale_approximation_params(scale_minus/inp_scale_max, mult_bits=16,
                                       force_shift_positive=self.force_shift_positive)
    do_scale, do_scale_type, do_shift, do_shift_type = \
        get_scale_approximation_params(local_rescale, mult_bits=q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)
    self.params["shift_value"] = [int(do_shift), int(do_shift_minus + g_eltwise_scale_bits)]
    self.params["shift_type"] = [do_shift_type, do_shift_minus_type]
    self.params["scale_value"] = [int(do_scale), int(scale0), int(scale1), int(do_scale_minus)]
    self.params["scale_type"] = [do_scale_type, Dtype.UINT16, Dtype.UINT16, do_scale_minus_type]


@op_register(OpType.SquaredDifference)
def squareddifference(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]

    x0 = inp0.betensor + (torch.tensor(0) if not self.quantized else torch.tensor(self.inputs[0].zerop))
    x1 = inp1.betensor + (torch.tensor(0) if not self.quantized else torch.tensor(self.inputs[1].zerop))

    if self.quantized:
        scale0, scale1, scale2, scale3 = self.params["scale_value"]
        shift0, shift1 = self.params["shift_value"]
        act_qmin, act_qmax = -2 ** 31, 2 ** 31 - 1
        Qy1 = linear_requantize((x0*scale1 - x1*scale2), scale3, shift1, 0, act_qmin, act_qmax)
        out.betensor = linear_requantize(Qy1*Qy1, scale0, shift0, out.zerop, out.qmin, out.qmax)
    else:
        minus = x0 - x1
        out.betensor = minus * minus
        if len(self.placeholders) < 1:
            ph0 = PyTensor(self.name+"/minus_outputs", minus.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            self.placeholders.append(ph0)
        self.placeholders[0].betensor = minus

    return out.betensor
