# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
import torch

# If x < -lambd, y = x + bias; If x > lambd, y = x - bias; Otherwise, y = 0

register_optype('SHRINK')


def shrink_func(inp, lambd, bias):
    output = torch.zeros_like(inp, device=inp.device)
    output[inp < -lambd] = inp[inp < -lambd] + bias
    output[inp > lambd] = inp[inp > lambd] - bias
    return output


@quant_register(OpType.SHRINK)
def shrink_quantize(self, *args):
    def shrink_lambda(x): return shrink_func(x, float(self.get_param("lambd")), float(self.get_param("bias")))
    self.attrs['lambda_func'] = shrink_lambda
    self.attrs['out_signed'] = True
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')

    self.params.pop('lambd')
    self.params.pop('bias')


@op_register(OpType.SHRINK)
def shrink(self, *args):
    def approximated_float_forward(self,  inp_tensor):
        if self.approximated and "lut" in self.constants:
            lut = self.constants["lut"].betensor
            out = lookup_float_index_lut(
                inp_tensor, lut, self.params['index_scale_value'], self.params['index_offset_value'], mirror_mode=True, value_offset_for_mirror_mode=self.params['value_offset_value'])
        else:
            lambd = float(self.get_param("lambd"))
            bias = float(self.get_param("bias"))
            out = shrink_func(inp_tensor, lambd, bias)
        return out
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self,  x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@approx_register(OpType.SHRINK)
def shrink_approx(self, *args):
    def set_min_max(inp, use_dynamic_lut, lambd):
        clip_min = 0
        if use_dynamic_lut:
            clip_max = max(abs(inp.min), abs(inp.max))
        else:
            clip_max = lambd + 2
        return clip_min, clip_max

    def shrink_lambda(x): return shrink_func(x, float(self.get_param("lambd")), float(self.get_param("bias")))

    self.attrs['set_min_max'] = lambda inp, use_dynamic_lut: set_min_max(
        inp, use_dynamic_lut, float(self.get_param("lambd")))
    self.attrs['lambda_func'] = shrink_lambda
    self.attrs['out_signed'] = False
    activation_module.unknown_approx(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('set_min_max')
    self.attrs.pop('out_signed')

    self.params['lut_mode'] = 'MIRROR'
