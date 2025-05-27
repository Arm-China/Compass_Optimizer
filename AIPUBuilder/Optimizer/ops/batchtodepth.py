# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_ERROR


@op_register(OpType.BatchToDepth)
def batchtodepth(self, *args):
    block_size_ = self.get_param('block_size')
    inp = self.inputs[0].betensor
    out = self.outputs[0].betensor
    n, h, w, c = inp.shape
    if self.inputs[0].ir_shape[0] != self.current_batch_size:
        OPT_ERROR("batch size in calibratoin or metric dataset should be equal to batch size in IR")
    y = inp.view(n // block_size_, block_size_, h, w, c).permute(0, 2, 3, 1, 4).contiguous()
    out = y.view(n // block_size_, h, w, c*block_size_)
    self.outputs[0].betensor = out
    return out


@quant_register(OpType.BatchToDepth)
def batchtodepth_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.qinvariant = inp.qinvariant
