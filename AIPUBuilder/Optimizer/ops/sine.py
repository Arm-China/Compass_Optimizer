# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.ops.cosine import trigonometric_quantize, trigonometric_forward, cosine_approx

import torch


@quant_register(OpType.Sine)
def sine_quantize(self, *args):
    trigonometric_quantize(self, torch.sin)


@approx_register(OpType.Sine)
def sine_approx(self, *args):
    cosine_approx(self, *args)


@op_register(OpType.Sine)
def sine(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    if self.quantized:
        q_bits_activation = inp.qbits
        if q_bits_activation > 8:
            x = inp.betensor.long() + inp.zerop
            lut = self.constants["lut"].betensor
            out.betensor = trigonometric_forward(self, x, lut)
        else:
            x = inp.betensor
            x = x - inp.qmin
            lut = self.constants["lut"].betensor
            x = torch.reshape(x, (-1,))
            y = lookup_lut_powerof2(x, lut, inp.qbits, False, dtype2bits(
                self.constants["lut"].dtype), is_signed(self.constants["lut"].dtype))
            out.betensor = torch.reshape(y, inp.betensor.shape)
    else:
        if self.approximated and "lut" in self.constants:
            lut = self.constants["lut"].betensor
            inp_tensor = inp.betensor.float()
            inter = (inp_tensor * (1/(2*torch.pi))).int()
            Fractional = inp_tensor - inter*2*torch.pi
            out.betensor = lookup_float_index_lut(
                Fractional, lut, self.params['index_scale_value'], self.params['index_offset_value'], mirror_mode=True, value_offset_for_mirror_mode=self.params['value_offset_value'])
        else:
            out.betensor = torch.sin(inp.betensor)
    return out.betensor
