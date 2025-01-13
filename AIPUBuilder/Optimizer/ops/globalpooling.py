# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.ops.pooling import pooling_quantize, pooling
import torch


@op_register(OpType.GlobalPool)
def globalpool_forward(self, *args):
    '''
    this op is used for ds forward to get the ds output shape when avgpool's output.ir_shape[1:3] == [1,1]

    :param self:
    :param args:
    :return:
    '''
    inp_shape = self.inputs[0].betensor.shape
    padding = (self.get_param('pad_left'),
               self.get_param('pad_right'),
               self.get_param('pad_top', optional=True, default_value=0),
               self.get_param('pad_bottom', optional=True, default_value=0))
    kernel_size = (inp_shape[1] + padding[2] + padding[3], inp_shape[2] + padding[0] + padding[1])
    self.params['kernel_y'] = kernel_size[0]
    self.params['kernel_x'] = kernel_size[1]

    out = pooling(self)
    return out


@quant_register(OpType.GlobalPool)
def globalpool_quantize(self, *args):
    pooling_quantize(self)
