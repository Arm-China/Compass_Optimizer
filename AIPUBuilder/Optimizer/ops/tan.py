# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
import torch


@quant_register(OpType.Tan)
def tan_quantize(self, *args):
    self.attrs['lambda_func'] = torch.tan
    self.attrs['out_signed'] = True
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')


@op_register(OpType.Tan)
def tan(self, *args):
    self.attrs['lambda_func'] = torch.tan
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@approx_register(OpType.Tan)
def tan_approx(self, *args):
    # By default, it is calculated directly on TPC
    self.params['is_perf_mode'] = False
