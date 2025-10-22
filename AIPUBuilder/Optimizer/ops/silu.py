# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module

import torch


@op_register(OpType.Silu)
def silu(self, *args):
    def approximated_float_forward(self,  inp_tensor):
        if self.approximated and "lut" in self.constants:
            lut = self.constants["lut"].betensor
            out = inp_tensor * lookup_float_index_lut(
                inp_tensor, lut, self.params['index_scale_value'], self.params['index_offset_value'], mirror_mode=True, value_offset_for_mirror_mode=self.params['value_offset_value'])
        else:
            out = torch.nn.functional.silu(inp_tensor)
        return out
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self,  x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@quant_register(OpType.Silu)
def silu_quantize(self, *args):
    def silu(x): return torch.nn.functional.silu(x)
    self.attrs['lambda_func'] = silu
    self.attrs['out_signed'] = True
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')


@approx_register(OpType.Silu)
def silu_approx(self, *args):
    def set_min_max(inp, use_dynamic_lut):
        if not use_dynamic_lut:
            clip_min = 0
            clip_max = 10
        else:
            clip_min = 0
            clip_max = max(abs(inp.min), abs(inp.max))
        return clip_min, clip_max

    def sigmoid_func(x): return torch.nn.functional.sigmoid(x * self.get_param('alpha', optional=True, default_value=1))

    if 'set_min_max' not in self.attrs:
        self.attrs['set_min_max'] = set_min_max
    self.attrs['lambda_func'] = sigmoid_func
    self.attrs['out_signed'] = False
    self.attrs['value_offset'] = -0.5
    activation_module.unknown_approx(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('set_min_max')
    self.attrs.pop('out_signed')
    self.attrs.pop('value_offset')
    if self.get_param('is_perf_mode', optional=True, default_value=False) and 'lut' in self.constants:
        self.params['lut_mode'] = 'MIRROR'
