# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *


register_optype('DeQuantize')


@quant_register(OpType.DeQuantize)
def dequantize_quant(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.qinvariant = inp.qinvariant
    out.qmin = inp.qmin
    out.qmax = inp.qmax


@op_register(OpType.DeQuantize)
def dequantize_forward(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.betensor = linear_dequantize(inp.betensor, inp.broadcast_scale, inp.broadcast_zerop)
    return out.betensor
