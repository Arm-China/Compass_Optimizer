# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.eltwise import eltwise_quantize, eltwise


@op_register(OpType.Add)
def add_forward(self, *args):
    self.params['method'] = 'ADD'
    eltwise(self, *args)
    self.params.pop('method')
    return self.outputs[0].betensor


@quant_register(OpType.Add)
def add_quantize(self, *args):
    self.params['method'] = 'ADD'
    eltwise_quantize(self, *args)
    self.params.pop('method')
