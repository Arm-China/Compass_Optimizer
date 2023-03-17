# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.ops.dilation2d import *
import torch

register_optype('Erosion')


@op_register(OpType.Erosion)
def erosion(self, *args):
    outp = dilation_erosion_fun(self, padding_value=float('inf'), compare_func=torch.amin, weight_reverse=True)
    self.outputs[0].betensor = outp
    return outp


@quant_register(OpType.Erosion)
def erosion_quantize(self, *args):
    dilation_quantize(self, *args)
