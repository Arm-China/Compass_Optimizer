# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *


@op_register(OpType.Negative)
def neg(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    if self.quantized:
        y = torch.neg(inp.betensor + inp.zerop) - out.zerop
        out.betensor = torch.clamp(y, out.qmin, out.qmax)
    else:
        out.betensor = torch.neg(inp.betensor)
    return out.betensor


@quant_register(OpType.Negative)
def neg_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    q_mode_activation = self.attrs["q_mode_activation"]
    out.qinvariant = inp.qinvariant
    out.scale = inp.scale
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.zerop = inp.zerop
    if QuantMode.is_asymmetric(q_mode_activation):
        if is_signed(inp.dtype):
            out.zerop = -1 * inp.zerop + 1
        else:
            out.dtype = bits2dtype(dtype2bits(inp.dtype), is_signed=True)
            out.qmin, out.qmax = dtype2range(out.dtype)
            out.zerop = -1 * inp.zerop - out.qmax
