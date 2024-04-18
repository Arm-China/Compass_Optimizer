# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch


@op_register(OpType.DataStride)
def datastride(self, *args):

    inpt = self.inputs[0].betensor
    inpt = nhwc2nchw(inpt)

    kernel_x = self.get_param('kernel_x')
    kernel_y = self.get_param('kernel_y')
    stride_x = self.get_param('stride_x')
    stride_y = self.get_param('stride_y')

    patches = inpt.unfold(2, kernel_y, stride_y).unfold(3, kernel_x, stride_x)
    patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()
    patches = patches.view(inpt.shape[0], -1, patches.shape[-2], patches.shape[-1])
    N, C, H, W = patches.size()
    bs = kernel_y
    patches = patches.view(N, bs, bs, C // (bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
    patches = patches.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
    patches = patches.view(N, C // (bs ** 2), H * bs, W * bs)  # (N, C//bs^2, H * bs, W * bs)
    patches = nchw2nhwc(patches)
    self.outputs[0].betensor = patches
    return patches


@quant_register(OpType.DataStride)
def quantize_datastride(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
