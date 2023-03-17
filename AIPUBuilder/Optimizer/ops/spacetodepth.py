# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch


def space_to_depth(x, bs):
    N, C, H, W = x.size()
    x = x.view(N, C, H // bs, bs, W // bs, bs)       # (N, C, H//bs, bs, W//bs, bs)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()     # (N, bs, bs, C, H//bs, W//bs)
    x = x.view(N, C * (bs ** 2), H // bs, W // bs)   # (N, C*bs^2, H//bs, W//bs)
    return x


@op_register(OpType.SpaceToDepth)
def spacetodepth(self, *args):
    block_size_x = self.get_param('block_size_x')
    block_size_y = self.get_param('block_size_y')
    if block_size_x != block_size_y:
        OPT_WARN("currently not support block_size_x != block_size_y in layer" +
                 self.attrs['layer_id'], op_name=str(self.type))
    inp = self.inputs[0].betensor
    # data format is NHWC
    inp = nhwc2nchw(inp)
    o = space_to_depth(inp, block_size_x)
    self.outputs[0].betensor = nchw2nhwc(o)
    return self.outputs[0].betensor


@quant_register(OpType.SpaceToDepth)
def spacetodepth_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.qinvariant = inp.qinvariant
