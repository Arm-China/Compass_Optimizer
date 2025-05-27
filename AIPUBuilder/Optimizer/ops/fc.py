# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.ops.conv import *
from AIPUBuilder.Optimizer.ops.activation import apply_with_activation
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch.nn as nn


@op_register(OpType.FullyConnected)
def fc(self, *args):
    inp, bias, weights = None, None, None
    if self.constants['weights'].dtype in [Dtype.FP32, Dtype.FP16]:
        inp = self.inputs[0].betensor.float()
        bias = self.constants["biases"].betensor.float()
        weights = self.constants["weights"].betensor.float()
    else:
        inp = self.inputs[0].betensor.double()
        bias = self.constants["biases"].betensor.clone().double()
        weights = self.constants["weights"].betensor.clone().double()
    aasrb = self.get_param('remain_shift',
                           optional=True, default_value=None)

    if self.quantized:
        # input's zerop has been absorbed to bias.
        # inp += self.inputs[0].zerop
        weights += self.constants["weights"].broadcast_zerop
        bias += self.constants['biases'].broadcast_zerop
        if aasrb is not None and (dtype2bits(self.constants["weights"].dtype) > 8 or dtype2bits(self.inputs[0].dtype) > 8):

            x = inp @ weights.T
            self.outputs[0].betensor = apply_with_activation(self, x,
                                                             *args, aasrb=(aasrb, bias))
            return self.outputs[0].betensor
    x = nn.functional.linear(inp, weights, bias,)
    self.outputs[0].betensor = apply_with_activation(self, x, *args)
    return self.outputs[0].betensor


@quant_register(OpType.FullyConnected)
def fc_quantize(self, *args):
    conv2d_quantize(self, *args)
    if 'remain_shift' in self.attrs:
        self.params['remain_shift'] = self.attrs['remain_shift']
