# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.bn import *
import torch

register_optype('BiasAdd')


@quant_register(OpType.BiasAdd)
def bias_add_quantize(self, *args):
    # bias_add is equal to batchnorm with weights == 1
    self.attrs["q_mode_weight"] = self.attrs["q_mode_activation"]
    self.attrs["q_mode_bias"] = self.attrs["q_mode_weight"]
    self.constants["weights"] = self.constants["weights_bk"]
    batch_norm_quantize(self, *args)
    self.constants.pop('weights_bk')
    self.constants.pop('weights')


@op_register(OpType.BiasAdd)
def bias_add_forward(self, *args):
    if not self.quantized:
        if 'weights_bk' not in self.constants.keys():
            self.constants["weights_bk"] = PyTensor(self.name + '/temp_weights_bk')
            self.constants["weights_bk"].betensor = torch.ones_like(self.constants["biases"].betensor)
            self.constants['weights_bk'].ir_shape = self.constants["biases"].shape
            self.constants['weights_bk'].ir_dtype = self.constants["biases"].ir_dtype
    self.constants["weights"] = PyTensor(self.name + '/temp_weights')
    self.constants["weights"].betensor = torch.ones_like(self.constants["biases"].betensor)
    self.constants['weights'].ir_shape = self.constants["biases"].shape
    self.constants['weights'].ir_dtype = self.constants["biases"].ir_dtype
    aflag = False
    if 'axis' not in self.params:
        aflag = True
        input_dim = self.inputs[0].betensor.dim()
        self.params['axis'] = input_dim - 1
    batch_norm(self, *args)
    if aflag:
        self.params.pop('axis')
    self.constants.pop('weights')
    return self.outputs[0].betensor
