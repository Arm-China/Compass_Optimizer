# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
import torch

register_optype('HardSigmoid')


def hardsigmoid_func(x, alpha, beta, clip_min, clip_max, dev):
    out = torch.maximum(torch.tensor(clip_min, device=dev),
                        torch.minimum(alpha * x + beta, torch.tensor(clip_max, device=dev)))
    return out


@quant_register(OpType.HardSigmoid)
def hardsigmoid_quantize(self, *args):
    alpha = self.get_param('alpha')
    beta = self.get_param('beta')
    # it will set optional to False when parser add 'clip_max' and 'clip_min' in future
    clip_max = self.get_param('clip_max', optional=True, default_value=1)
    clip_min = self.get_param('clip_min', optional=True, default_value=0)
    dev = self.inputs[0].betensor.device

    self.attrs['lambda_func'] = lambda x: hardsigmoid_func(x, alpha, beta, clip_min, clip_max, dev)
    self.attrs['out_signed'] = clip_min < 0 or self.force_dtype_int
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')


@op_register(OpType.HardSigmoid)
def hardsigmoid(self, *args):
    def approximated_float_forward(self,  inp_tensor):
        dev = self.inputs[0].betensor.device
        alpha = self.get_param('alpha')
        beta = self.get_param('beta')
        # it will set optional to False when parser add 'clip_max' and 'clip_min' in future
        clip_max = self.get_param('clip_max', optional=True, default_value=1)
        clip_min = self.get_param('clip_min', optional=True, default_value=0)
        out = hardsigmoid_func(inp_tensor, alpha, beta, clip_min, clip_max, dev)
        return out
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self,  x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


def hardsigmoid_out_signed(self):
    clip_min = self.get_param('clip_min', optional=True, default_value=0)
    return False if clip_min >= 0 else True


@approx_register(OpType.HardSigmoid)
def hardsigmoid_approx(self, *args):
    # By default, it is calculated directly on AIFF
    self.params['is_perf_mode'] = True
