# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.eltwise import eltwise_quantize, eltwise


@op_register(OpType.Mul)
def mul_forward(self, *args):
    self.params['method'] = 'MUL'
    eltwise(self, *args)
    self.params.pop('method')
    return self.outputs[0].betensor


@quant_register(OpType.Mul)
def mul_quantize(self, *args):
    self.params['method'] = 'MUL'
    eltwise_quantize(self, *args)
    self.params.pop('method')
