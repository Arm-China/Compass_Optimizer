# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.stridedslice import *


@op_register(OpType.Slice)
def slice_forward(self, *args):
    return stridedslice(self, *args)


@quant_register(OpType.Slice)
def slice_quantize(self, *args):
    stridedslice_quantize(self, *args)
