# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *


register_optype('Quantize')


@quant_register(OpType.Quantize)
def quantize_quant(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.qinvariant = inp.qinvariant
    out.qmin = inp.qmin
    out.qmax = inp.qmax


@op_register(OpType.Quantize)
def quantize_forward(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    if out.qmin is None:
        out.betensor = inp.betensor
    else:
        out.betensor = linear_quantize_clip(inp.betensor, out.scale, out.zerop, out.qmin, out.qmax)
    return out.betensor
