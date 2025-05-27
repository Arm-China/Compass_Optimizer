# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
import torch

# y = x for x > alpha, y = 0 otherwise, is applied to the tensor elementwise.

register_optype('THRESHOLDEDRELU')


@quant_register(OpType.THRESHOLDEDRELU)
def thresholdedrelu_quantize(self, *args):
    alpha = float(self.get_param("alpha"))
    self.attrs['lambda_func'] = lambda x: torch.nn.functional.threshold(x, alpha, 0)
    self.attrs['out_signed'] = True if alpha < 0.0 else False
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')


@op_register(OpType.THRESHOLDEDRELU)
def thresholdedrelu(self, *args):
    def float_forward(self,  inp_tensor):
        alpha = float(self.get_param("alpha"))
        out = torch.nn.functional.threshold(inp_tensor, alpha, 0)
        return out
    self.attrs['lambda_func'] = lambda x: float_forward(self,  x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@approx_register(OpType.THRESHOLDEDRELU)
def thresholdrelu_approx(self, *args):
    # By default, it is calculated directly on AIFF
    self.params['is_perf_mode'] = True


def threshold_out_signed(self):
    alpha = float(self.get_param("alpha"))
    return False if alpha >= 0 else True
