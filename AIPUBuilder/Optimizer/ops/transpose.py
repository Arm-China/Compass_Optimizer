# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.permute import *


@op_register(OpType.Transpose)
def transpose(self, *args):
    return permute(self, *args)


@quant_register(OpType.Transpose)
def transpose_quantize(self, *args):
    permute_quantize(self, *args)
