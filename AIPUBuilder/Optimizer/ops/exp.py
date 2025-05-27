# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.ops.softmax import softmax_approx
import AIPUBuilder.Optimizer.ops.activation as activation_module
import torch


@quant_register(OpType.Exp)
def exp_quantize(self, *args):
    self.attrs['lambda_func'] = torch.exp
    self.attrs['out_signed'] = False or self.force_dtype_int
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')


@op_register(OpType.Exp)
def exp(self, *args):
    def approximated_float_forward(self,  inp_tensor):
        if self.approximated and "lut" in self.constants:
            lut = self.constants["lut"].betensor
            f_vdata = inp_tensor * 1.442695
            out = x3_aiff_exp_approximation(f_vdata, lut)
        else:
            out = torch.exp(inp_tensor)
        return out
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self,  x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@approx_register(OpType.Exp)
def elu_approx(self, *args):
    softmax_approx(self, *args)
