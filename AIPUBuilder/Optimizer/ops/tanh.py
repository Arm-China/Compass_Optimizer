# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch

register_optype('Tanh')


@quant_register(OpType.Tanh)
def tanh_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    out = self.outputs[0]
    out.qbits = q_bits_activation
    out_sign = True
    dev = inp.betensor.device
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, out_sign)
    iqmin, iqmax = dtype2range(inp.dtype)
    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut = linear_dequantize(torch.linspace(iqmin, iqmax, steps=lsteps), inp.scale, inp.zerop)
    lut = torch.tanh(lut)
    lut = linear_quantize_clip(lut, out.scale, out.zerop, out.qmin, out.qmax)
    self.constants["lut"] = PyTensor(self.name+"/tanh_lut", lut.cpu().numpy().astype(dtype2nptype(out.dtype)))
    out.qinvariant = False


@op_register(OpType.Tanh)
def tanh(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    if self.quantized:
        lut_in_bits = inp.qbits
        lut_out_bits = out.qbits
        in_is_signed = is_signed(inp.dtype)
        out_is_signed = is_signed(out.dtype)
        lut = self.constants["lut"].betensor
        x = inp.betensor
        y = lookup_lut_powerof2(x, lut, lut_in_bits, in_is_signed, lut_out_bits, out_is_signed)
        out.betensor = torch.reshape(y, inp.betensor.shape)
    else:
        out.betensor = torch.tanh(inp.betensor)
    return out.betensor
