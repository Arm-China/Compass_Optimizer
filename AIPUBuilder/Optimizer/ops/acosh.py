# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
import torch

# acosh(x) = log(x + sqrt(x^2-1)) x∈[1，inf)， y∈[0，inf)
# if x < 1, y is nan


@quant_register(OpType.Acosh)
def acosh_quantize(self, *args):
    self.attrs['lambda_func'] = torch.acosh
    self.attrs['out_signed'] = False
    activation_module.unknown_quantize(self, *args)
    for k in ['lambda_func', 'out_signed']:
        self.attrs.pop(k)


@op_register(OpType.Acosh)
def acosh(self, *args):
    def approximated_float_forward(self,  inp_tensor):
        if self.approximated and "lut" in self.constants:
            lut = self.constants["lut"].betensor
            out = lookup_float_index_lut(inp_tensor, lut,
                                         self.params['index_scale_value'],
                                         self.params['index_offset_value'],
                                         mirror_mode=False,
                                         value_offset_for_mirror_mode=self.params['value_offset_value'])
        else:
            out = torch.acosh(inp_tensor)
        return out
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self,  x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@approx_register(OpType.Acosh)
def acosh_approx(self, *args):
    def set_min_max(inp, use_dynamic_lut):
        if not use_dynamic_lut:
            clip_min = 1
            clip_max = 256
        else:
            clip_min = inp.min
            clip_max = inp.max
        return clip_min, clip_max

    def acosh(x):
        x = torch.clamp(x, min=1)
        return torch.acosh(x)
    self.attrs['set_min_max'] = set_min_max
    self.attrs['lambda_func'] = acosh
    self.attrs['out_signed'] = False
    activation_module.unknown_approx(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('set_min_max')
    self.attrs.pop('out_signed')
