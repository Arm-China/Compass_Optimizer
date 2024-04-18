# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.ops.conv import *
from AIPUBuilder.Optimizer.ops.activation import apply_with_activation
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch.nn as nn


@op_register(OpType.FullyConnected)
def fc(self, *args):
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
        if aasrb is not None:
            bias = aiff_clear_lower_bits_for_bias(bias, self)
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
