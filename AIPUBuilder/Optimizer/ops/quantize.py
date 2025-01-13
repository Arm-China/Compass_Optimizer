# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

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
    round_mode = self.get_param('round_mode', optional=True, default_value="ROUND_TO_EVEN")
    round_func = get_round_mode_func(round_mode)
    if out.qmin is None:
        out.betensor = inp.betensor
    else:
        out.betensor = linear_quantize_clip(inp.betensor, out.broadcast_scale,
                                            out.broadcast_zerop, out.qmin, out.qmax, round_func=round_func)
    return out.betensor
