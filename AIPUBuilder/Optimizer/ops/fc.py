# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.ops.conv import *
from AIPUBuilder.Optimizer.ops.activation import apply_with_activation
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch.nn as nn


@op_register(OpType.FullyConnected)
def fc(self, *args):
    inp = self.inputs[0].betensor.float()
    bias = self.constants["biases"].betensor.clone().float()
    weights = self.constants["weights"].betensor.clone().float()
    if self.quantized:
        # input's zerop has been absorbed to bias.
        # inp += self.inputs[0].zerop
        w_zp = self.constants["weights"].zerop
        w_zshape = [1] * weights.dim()
        w_zshape[0] = -1
        weights += w_zp.reshape(w_zshape) if isinstance(w_zp, torch.Tensor) else w_zp
        bias += self.constants['biases'].zerop
    x = nn.functional.linear(inp,
                             weights,
                             bias,
                             )
    requant_scale = 1
    requant_shift = 0
    if self.quantized:
        if 'scale_value' in self.params:
            requant_scale = self.params['scale_value']
        elif "scale" in self.constants:
            requant_scale = self.constants["scale"].betensor

        if 'shift_value' in self.params:
            requant_shift = self.params['shift_value']
        elif "shift" in self.constants:
            requant_shift = self.constants["shift"].betensor

    x = apply_with_activation(self, x,
                              self.inputs[0].scale * self.constants["weights"].scale, 0,
                              requant_scale,
                              requant_shift,
                              *args)

    self.outputs[0].betensor = x
    return x


@quant_register(OpType.FullyConnected)
def fc_quantize(self, *args):
    conv2d_quantize(self, *args)
