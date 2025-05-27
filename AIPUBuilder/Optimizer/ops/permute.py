# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

import torch


@op_register(OpType.Permute)
def permute(self, *args):
    inp = self.inputs[0].betensor
    out = self.outputs[0]
    perm = self.get_param('perm')
    if not isinstance(perm, list):
        perm = list(reversed([i for i in range(len(inp.shape))]))
    out.betensor = inp.permute(perm)
    return out.betensor


@quant_register(OpType.Permute)
def permute_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qmin, out.qmax = inp.qmin, inp.qmax
    out.qinvariant = inp.qinvariant
