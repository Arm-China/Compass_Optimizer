# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
import torch
import math


#  0.5 * x * (1 + tanh ( sqrt(2/pi) * (x + 0.044715 * x**3)  ) )
#  0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def phi_tanh_approx(x):
    #  increase compatibility for 3.6.5(no torch.pi api)
    return 0.5 * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / math.pi)) * (x + 0.044715 * torch.pow(x, 3))))


def gelu_tanh_approx(x):
    return x * phi_tanh_approx(x)


def float_gelu(self,  x):
    apprx = self.get_param('approximate').lower()
    if apprx == 'none':
        torch_gelu = torch.nn.GELU()
        x = torch_gelu(x)
    elif apprx == 'tanh':
        x = gelu_tanh_approx(x)
    return x


def approximated_float_forward(self,  inp_tensor):
    apprx = self.get_param('approximate').lower()
    if self.approximated and "lut" in self.constants:
        lut = self.constants["lut"].betensor
        out = inp_tensor * lookup_float_index_lut(
            inp_tensor, lut, self.params['index_scale_value'], self.params['index_offset_value'], mirror_mode=True, value_offset_for_mirror_mode=self.params['value_offset_value'])
    else:
        if apprx == 'none':
            out = torch.nn.functional.gelu(inp_tensor)
        elif apprx == 'tanh':
            out = gelu_tanh_approx(inp_tensor)
    return out


@quant_register(OpType.GELU)
def gelu_quantize(self, *args):
    def gelu(x): return float_gelu(self, x)
    self.attrs['lambda_func'] = gelu
    self.attrs['out_signed'] = True
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')


@approx_register(OpType.GELU)
def gelu_approx(self, *args):

    def set_min_max(inp, use_dynamic_lut):
        if not use_dynamic_lut:
            clip_min = 0
            clip_max = 6
        else:
            clip_min = 0
            clip_max = max(abs(inp.min), abs(inp.max))
        return clip_min, clip_max

    self.attrs['set_min_max'] = set_min_max
    self.attrs['lambda_func'] = phi_tanh_approx
    self.attrs['out_signed'] = False
    self.attrs['value_offset'] = -0.5
    activation_module.unknown_approx(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('set_min_max')
    self.attrs.pop('out_signed')
    self.attrs.pop('value_offset')

    self.params['lut_mode'] = 'MIRROR'


@op_register(OpType.GELU)
def gelu(self, *args):
    apprx = self.get_param('approximate').lower()  # NONE, Tanh
    if apprx not in ['none', 'tanh']:
        OPT_ERROR('GELU dont support approximation method:%s' % apprx)
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self, x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')

    return self.outputs[0].betensor
