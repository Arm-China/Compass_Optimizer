# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.ops.eltwise import eltwise_quantize, eltwise


@op_register(OpType.Sub)
def sub_forward(self, *args):
    self.params['method'] = 'SUB'
    self.params['with_activation'] = 'NONE'
    eltwise(self, *args)
    self.params.pop('method')
    self.params.pop('with_activation')

    return self.outputs[0].betensor


@quant_register(OpType.Sub)
def sub_quantize(self, *args):
    self.params['method'] = 'SUB'
    self.params['with_activation'] = 'NONE'
    eltwise_quantize(self, *args)
    self.params.pop('method')
    self.params.pop('with_activation')
