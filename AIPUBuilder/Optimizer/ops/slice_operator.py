# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.stridedslice import *


@op_register(OpType.Slice)
def slice_forward(self, *args):
    return stridedslice(self, *args)


@quant_register(OpType.Slice)
def slice_quantize(self, *args):
    stridedslice_quantize(self, *args)
