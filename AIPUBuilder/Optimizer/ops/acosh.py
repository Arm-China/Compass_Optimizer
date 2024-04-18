# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch

# acosh(x) = log(x + sqrt(x^2-1)) x∈[1，inf)， y∈[0，inf)
# if x < 1, y is nan


@quant_register(OpType.Acosh)
def acosh_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    out = self.outputs[0]
    if inp.extrema_min < 1:
        OPT_WARN("input of Acosh(layer_id=%s) must be >= 1, otherwise the output is nan, please check!"
                 % (self.attrs['layer_id']))
    out.qbits = q_bits_activation
    out_sign = False or self.force_dtype_int
    dev = inp.betensor.device
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, out_sign)
    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut = linear_dequantize(torch.linspace(inp.qmin, inp.qmax, steps=lsteps, device=dev), inp.scale, inp.zerop)
    lut = torch.acosh(lut)
    lut = linear_quantize_clip(lut, out.scale, out.zerop, out.qmin, out.qmax)
    self.constants["lut"] = PyTensor(self.name+"/acosh_lut", lut.cpu().numpy().astype(dtype2nptype(out.dtype)))
    out.qinvariant = False


@op_register(OpType.Acosh)
def acosh(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    if self.quantized:
        x = inp.betensor
        x = x - inp.qmin
        lut = self.constants["lut"].betensor
        x = torch.reshape(x, (-1,))
        y = lookup_lut_powerof2(x, lut, inp.qbits, False, dtype2bits(
            self.constants["lut"].dtype), is_signed(self.constants["lut"].dtype))
        out.betensor = torch.reshape(y, inp.betensor.shape)
    else:
        out.betensor = torch.acosh(inp.betensor)
        if torch.any(torch.isnan(out.betensor)):
            out.betensor = torch.where(torch.isnan(out.betensor), torch.zeros_like(inp.betensor), out.betensor)
            OPT_WARN('layer_id=%s, type=%s, the output has nan, please confirm whether input is >= 1, now set nan to zero'
                     % (self.attrs['layer_id'], str(self.type)))
    return out.betensor
