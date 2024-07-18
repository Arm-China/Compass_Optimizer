# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
from AIPUBuilder.Optimizer.ops.silu import silu_approx
from AIPUBuilder.Optimizer.utils.math_utils import *
import torch

register_optype('Sigmoid')


@op_register(OpType.Sigmoid)
def sigmoid_forward(self, *args):
    def approximated_float_forward(self,  inp_tensor):
        if self.approximated and "lut" in self.constants:
            lut = self.constants["lut"].betensor
            out = lookup_float_index_lut(inp_tensor, lut,
                                         self.params['index_scale_value'],
                                         self.params['index_offset_value'],
                                         mirror_mode=True,
                                         value_offset_for_mirror_mode=self.params['value_offset_value'])
        else:
            out = torch.sigmoid(inp_tensor)
        return out
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self,  x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@quant_register(OpType.Sigmoid)
def sigmoid_quantize(self, *args):
    def sigmoid_func(x): return torch.sigmoid(x)
    self.attrs['lambda_func'] = sigmoid_func
    self.attrs['out_signed'] = False or self.force_dtype_int
    offset = 0.0
    if self.type in [OpType.BasicLSTM, OpType.GRUv3, OpType.GRUv1]:
        offset = torch.sigmoid(torch.tensor(0.0)).item()
    self.attrs['mirror_offset'] = offset

    activation_module.unknown_quantize(self, *args)

    for k in ['lambda_func', 'out_signed', 'mirror_offset']:
        self.attrs.pop(k)


@approx_register(OpType.Sigmoid)
def sigmoid_approx(self, *args):
    silu_approx(self, *args)
