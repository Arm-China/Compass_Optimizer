# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.ops.silu import silu_approx
import AIPUBuilder.Optimizer.ops.activation as activation_module

import torch

register_optype('Swish')


def swish_func(x, alpha):
    return x * torch.sigmoid(alpha * x)


@quant_register(OpType.Swish)
def swish_quantize(self, *args):
    self.attrs['lambda_func'] = lambda x: swish_func(x,  self.get_param('alpha'))
    self.attrs['out_signed'] = True
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')


@op_register(OpType.Swish)
def swish(self, *args):
    def approximated_float_forward(self,  inp_tensor):
        if self.approximated and "lut" in self.constants:
            lut = self.constants["lut"].betensor
            out = inp_tensor * lookup_float_index_lut(
                inp_tensor, lut, self.params['index_scale_value'], self.params['index_offset_value'], mirror_mode=True, value_offset_for_mirror_mode=self.params['value_offset_value'])
        else:
            out = swish_func(inp_tensor,  self.get_param('alpha'))
        return out
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self,  x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@approx_register(OpType.Swish)
def swish_approx(self, *args):
    silu_approx(self, *args)
