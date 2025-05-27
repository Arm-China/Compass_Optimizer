# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.upsamplebyindex import upsamplebyindex_quantize, upsamplebyindex


@quant_register(OpType.MaxUnpool)
def maxunpool_quantize(self, *args):
    upsamplebyindex_quantize(self, *args)


@op_register(OpType.MaxUnpool)
def maxunpool(self, *args):
    upsamplebyindex(self, *args)
