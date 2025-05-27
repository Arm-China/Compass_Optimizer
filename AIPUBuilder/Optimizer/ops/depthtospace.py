# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.logger import OPT_WARN
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch


@op_register(OpType.DepthToSpace)
def depthtospace(self, *args):
    block_size_x = self.get_param('block_size_x')
    block_size_y = self.get_param('block_size_y')
    mode = self.get_param('mode', optional=True, default_value='DCR').upper()
    if block_size_x != block_size_y:
        OPT_WARN("currently not support block_size_x != block_size_y in layer" +
                 self.attrs['layer_id'], op_name=str(self.type))
    inp = self.inputs[0].betensor
    out = self.outputs[0].betensor
    if inp.dim() != 4:
        OPT_FATAL("currently only support 4 dims input in layer" +
                  self.attrs['layer_id'], op_name=str(self.type))
    # data format is NHWC
    N, H, W, C = inp.size()
    new_channel = C // (block_size_x * block_size_y)
    if mode == 'DCR':
        x = inp.view(N, H, W, block_size_y, block_size_x, new_channel)  # (N, H, W, bs, bs, C//bs^2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (N, H, bs, W, bs, C//bs^2)
    elif mode == 'CRD':
        x = inp.view(N, H, W, new_channel, block_size_y, block_size_x)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()  # (N, H, bs, W, bs, C//bs^2)
    else:
        OPT_FATAL("unsupported mode: %s for DepthToSpace in node:%s" % (mode, self.name))
    out = x.view(N, H * block_size_y, W * block_size_x, new_channel)
    self.outputs[0].betensor = out
    return out


@quant_register(OpType.DepthToSpace)
def depthtospace_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.qinvariant = inp.qinvariant
