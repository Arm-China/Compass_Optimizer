# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
import torch

register_optype('Hardswish')


@quant_register(OpType.Hardswish)
def hardswish_quantize(self, *args):
    self.attrs['lambda_func'] = torch.nn.functional.hardswish
    self.attrs['out_signed'] = True
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')


@op_register(OpType.Hardswish)
def hardswish(self, *args):
    self.attrs['lambda_func'] = torch.nn.functional.hardswish
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@approx_register(OpType.Hardswish)
def hardswish_approx(self, *args):
    # By default, it is calculated directly on AIFF
    self.params['is_perf_mode'] = True
