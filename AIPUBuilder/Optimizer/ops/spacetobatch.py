# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch


@op_register(OpType.SpaceToBatch)
def spacetobatch(self, *args):
    block_size_x = self.get_param('block_size_x')
    block_size_y = self.get_param('block_size_y')
    pad_left = self.get_param('pad_left')
    pad_right = self.get_param('pad_right')
    pad_top = self.get_param('pad_top')
    pad_bottom = self.get_param('pad_bottom')
    inp = self.inputs[0].betensor
    out = self.outputs[0].betensor
    if self.inputs[0].shape[0] != inp.shape[0]:
        OPT_ERROR("batch size in calibratoin or metric dataset should be equal to batch size in IR")
    # inp is NHWC format
    paddings = (0, 0, pad_left, pad_right, pad_top, pad_bottom)
    # TODO: support per-channel zerop and pad the per-channel zerop
    pad_value = -self.inputs[0].zerop[0] if self.quantized else 0
    y = torch.nn.functional.pad(inp, paddings, value=pad_value)
    n, h, w, c = y.shape
    y = y.view(n, h//block_size_y, block_size_y, w//block_size_x, block_size_x, c)
    y = y.permute(2, 4, 0, 1, 3, 5).contiguous()
    out = y.view(n*block_size_x*block_size_y, h//block_size_y, w//block_size_x, c)
    self.outputs[0].betensor = out
    return out


@quant_register(OpType.SpaceToBatch)
def spacetobatch_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.qinvariant = inp.qinvariant
