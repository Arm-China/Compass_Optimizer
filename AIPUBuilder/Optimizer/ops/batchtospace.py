# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_ERROR


@op_register(OpType.BatchToSpace)
def batchtospace(self, *args):
    block_size_x = self.get_param('block_size_x')
    block_size_y = self.get_param('block_size_y')
    crop_left = self.get_param('crop_left')
    crop_right = self.get_param('crop_right')
    crop_top = self.get_param('crop_top')
    crop_bottom = self.get_param('crop_bottom')

    inp = self.inputs[0].betensor
    out = self.outputs[0].betensor
    n, h, w, c = inp.shape
    if self.inputs[0].ir_shape[0] != inp.shape[0]:
        OPT_ERROR("batch size in calibratoin or metric dataset should be equal to batch size in IR")
    y = inp.view(block_size_y, block_size_x, n // (block_size_x*block_size_y), h, w, c)
    y = y.permute(2, 3, 0, 4, 1, 5).contiguous()
    y = y.view(n // (block_size_x*block_size_y), h*block_size_y, w*block_size_x, c)
    out = y[:, crop_top:h*block_size_y-crop_bottom, crop_left:w*block_size_x-crop_right, :]
    self.outputs[0].betensor = out
    return out


@quant_register(OpType.BatchToSpace)
def batchtospace_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.qinvariant = inp.qinvariant
