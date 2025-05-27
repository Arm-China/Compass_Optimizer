# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
import torch


@quant_register(OpType.Square)
def square_quantize(self, *args):
    self.attrs['lambda_func'] = torch.square
    self.attrs['out_signed'] = False or self.force_dtype_int
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')


@op_register(OpType.Square)
def square(self, *args):
    self.attrs['lambda_func'] = torch.square
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@approx_register(OpType.Square)
def square_approx(self, *args):
    # By default, it is calculated directly on AIFF
    self.params['is_perf_mode'] = True
