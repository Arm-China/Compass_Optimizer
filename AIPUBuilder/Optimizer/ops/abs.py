# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *


@op_register(OpType.Abs)
def Abs_forward(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    if self.quantized:
        out.betensor = torch.clamp(torch.abs(inp.betensor + inp.zerop), out.qmin, out.qmax)
    else:
        out.betensor = torch.abs(inp.betensor)

    return out.betensor


@quant_register(OpType.Abs)
def abs_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    q_mode_activation = self.attrs["q_mode_activation"]
    out.qinvariant = inp.qinvariant
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = 0
    out.dtype = bits2dtype(out.qbits, is_signed=False)
    out.qmin, out.qmax = dtype2range(out.dtype)
