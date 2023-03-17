# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch


@op_register(OpType.Squeeze)
def squeeze(self, *args):
    axis = self.get_param('axis')
    inp = self.inputs[0].betensor
    out = self.outputs[0].betensor
    out = torch.squeeze(inp, dim=axis)
    self.outputs[0].betensor = out
    return out


@quant_register(OpType.Squeeze)
def squeeze_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.qinvariant = inp.qinvariant
