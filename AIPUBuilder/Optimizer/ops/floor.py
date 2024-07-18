# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
from AIPUBuilder.Optimizer.utils import *
import torch


@op_register(OpType.Floor)
def floor(self, *args):
    self.attrs['lambda_func'] = torch.floor
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@quant_register(OpType.Floor)
def floor_quantize(self, *args):
    self.attrs['lambda_func'] = torch.floor
    self.attrs['out_signed'] = True
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')


@approx_register(OpType.Floor)
def floor_approx(self, *args):
    # this is not currently used because it is the same as the float process
    self.params['is_perf_mode'] = False
