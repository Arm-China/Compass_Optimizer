# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch

# Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise

register_optype('Hardmax')


@quant_register(OpType.Hardmax)
def Hardmax_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    out = self.outputs[0]

    out.scale = 1
    out.zerop = 0
    out.dtype = inp.dtype
    out.qbits = inp.qbits
    out.qmin = inp.qmin
    out.qmax = inp.qmax
    out.qinvariant = True


@op_register(OpType.Hardmax)
def hardmax(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]

    axis = int(self.get_param('axis'))  # only int

    argx = torch.argmax(inp.betensor, axis, keepdim=True)
    output = None
    for ax in range(inp.betensor.shape[axis]):
        tmp_input = torch.index_select(inp.betensor, axis, torch.tensor([ax], device=inp.betensor.device))
        tmp_index = torch.full(tmp_input.shape, ax, device=inp.betensor.device)
        tmp_output = torch.where(tmp_index == argx, torch.ones_like(tmp_input), torch.zeros_like(tmp_input))
        output = tmp_output if output == None else torch.cat((output, tmp_output), axis)
    out.betensor = output

    return out.betensor
