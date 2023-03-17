# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch

# If x < -lambd, y = x + bias; If x > lambd, y = x - bias; Otherwise, y = 0

register_optype('SHRINK')


def shrink_func(inp, lambd, bias):
    output = torch.zeros_like(inp, device=inp.device)
    output[inp < -lambd] = inp[inp < -lambd] + bias
    output[inp > lambd] = inp[inp > lambd] - bias
    return output


@quant_register(OpType.SHRINK)
def shrink_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    out = self.outputs[0]

    lambd = float(self.get_param("lambd"))
    bias = float(self.get_param("bias"))

    out.qbits = q_bits_activation
    out_sign = True
    dev = inp.betensor.device
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, out_sign)
    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut = linear_dequantize(torch.linspace(inp.qmin, inp.qmax, steps=lsteps, device=dev), inp.scale, inp.zerop)
    lut = shrink_func(lut, lambd, bias)
    lut = linear_quantize_clip(lut, out.scale, out.zerop, out.qmin, out.qmax)
    self.constants["lut"] = PyTensor(self.name+"/shrink_lut", lut.cpu().numpy().astype(dtype2nptype(out.dtype)))
    out.qinvariant = False

    self.params.pop('lambd')
    self.params.pop('bias')


@op_register(OpType.SHRINK)
def shrink(self, *args):
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
        lambd = float(self.get_param("lambd"))
        bias = float(self.get_param("bias"))
        out.betensor = shrink_func(inp.betensor, lambd, bias)

    return out.betensor
