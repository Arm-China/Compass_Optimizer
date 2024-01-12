# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch


@op_register(OpType.MatMul)
def matmul_forward(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]
    x = inp0.betensor.float()
    y = inp1.betensor.float()
    aasrb = self.get_param('remain_shift',
                           optional=True, default_value=None)
    if self.get_param('trans_a'):
        if x.dim() == 0:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.transpose(-1, -2)
    if self.get_param('trans_b'):
        if y.dim() == 0:
            y = y.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 1:
            y = y.unsqueeze(0)
        y = y.transpose(-1, -2)
    if self.quantized:
        x = x + inp0.zerop
        y = y + inp1.zerop
    z = torch.matmul(x, y)
    if self.quantized:
        do_scale = self.params["scale_value"]
        do_shift = self.params['shift_value']
        if aasrb is not None:
            z, do_shift = aiff_ahead_shift_bias(z, do_shift, None, int(aasrb))
        z = linear_requantize(z, do_scale, do_shift, out.zerop, out.qmin, out.qmax)
    else:
        if len(self.placeholders) < 1:
            ph0 = PyTensor(self.name+"/matmul_result", z.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            self.placeholders.append(ph0)
        self.placeholders[0].betensor = z
        z *= self.get_param('fused_multiplier', optional=True, default_value=1)
    out.betensor = z
    return out.betensor


@quant_register(OpType.MatMul)
def matmul_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_activation = self.attrs["q_bits_activation"]
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]
    out_signed = is_signed(inp0.dtype) or is_signed(inp1.dtype)
    out.qinvariant = inp0.qinvariant and inp1.qinvariant
    if out.qinvariant:
        out.scale = 1.0
        out.zerop = 0
        out.qbits, out.dtype = range2dtype(out.extrema_min, out.extrema_max, force_int=self.force_dtype_int)
    else:
        out.qbits = q_bits_activation
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, out.qbits, is_signed=out_signed)
    out_scale = out.scale
    if 'fused_multiplier' in self.params:
        fused_multiplier = self.params['fused_multiplier']
        self.params.pop('fused_multiplier')
        plh = self.placeholders[0]
        plh.qbits = q_bits_activation
        plh.scale, plh.zerop, plh.qmin, plh.qmax, plh.dtype = get_linear_quant_params_from_tensor(
            plh, q_mode_activation, plh.qbits, is_signed=out_signed)
        out_scale = (1.0 if out.qinvariant else plh.scale) * fused_multiplier
    do_scale, do_scale_type, do_shift, do_shift_type = \
        get_scale_approximation_params(out_scale / (inp0.scale * inp1.scale),
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)
    self.params["shift_value"] = int(do_shift)
    self.params["shift_type"] = do_shift_type
    self.params["scale_value"] = int(do_scale)
    self.params["scale_type"] = do_scale_type
    if 'remain_shift' in self.attrs:
        self.params['remain_shift'] = self.attrs['remain_shift']
