# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *

register_optype('NoOp')


@op_register(OpType.NoOp)
def noop(self, *args):
    for ot in self.outputs:
        ot.betensor = self.inputs[0].betensor
    return [ot.betensor for ot in self.outputs]


@quant_register(OpType.NoOp)
def noop_quantize(self, *args):
    inp = self.inputs[0]
    for ot in self.outputs:
        ot.clone_qinfo(inp)
