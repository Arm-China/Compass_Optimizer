# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.ops.conv import conv2d, conv2d_quantize


@op_register(OpType.DepthwiseConv)
def depthwise_conv2d(self, *args):
    x = conv2d(self, *args)
    return x


@quant_register(OpType.DepthwiseConv)
def depthwise_conv2d_quantize(self, *args):
    conv2d_quantize(self, *args)
