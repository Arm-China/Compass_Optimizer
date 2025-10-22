# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import torch

register_optype('Shape')


@op_register(OpType.Shape)
def shape_forward(self, *args):
    inp = self.inputs[0].betensor
    start = self.get_param('start')
    end = self.get_param('end')
    out = torch.tensor(inp.shape[start:end], device=inp.device)
    self.outputs[0].betensor = out
    return out


@quant_register(OpType.Shape)
def shape_quantize(self, *args):
    # q_bits_activation = self.attrs["q_bits_activation"]
    out = self.outputs[0]
    out.scale = 1.
    out.zerop = 0
    out.qbits = 32
    out.dtype = Dtype.INT32
    out.qinvariant = True
