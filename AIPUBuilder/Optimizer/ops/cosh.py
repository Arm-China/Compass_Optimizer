# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.ops.sinh import sinh_approx
import AIPUBuilder.Optimizer.ops.activation as activation_module
import torch

# y = cosh(x) = (e^x + e^-x)/2 x∈R， y∈(1,inf)


def cosh_func(x, offset=80.0):
    if not isinstance(offset, torch.Tensor):
        offset = torch.tensor(offset, dtype=x.dtype, device=x.device)

    output = torch.cosh(x)
    # CP-24165,avoid inf
    negetive_x_ge_offset_mask = x < -offset
    positive_x_ge_offset_mask = x > offset
    offset_exp = torch.exp(offset) / 2
    if True in negetive_x_ge_offset_mask:
        negetive_x_ge_offset_value = x[negetive_x_ge_offset_mask]
        negetive_x_ge_offset_output = torch.exp(negetive_x_ge_offset_value) / 2 + \
            offset_exp * torch.exp(torch.abs(negetive_x_ge_offset_value + offset))
        output[negetive_x_ge_offset_mask] = negetive_x_ge_offset_output
    if True in positive_x_ge_offset_mask:
        positive_x_ge_offset_value = x[positive_x_ge_offset_mask]
        positive_x_ge_offset_output = offset_exp * \
            torch.exp(positive_x_ge_offset_value - offset) + torch.exp(torch.neg(positive_x_ge_offset_value)) / 2
        output[positive_x_ge_offset_mask] = positive_x_ge_offset_output
    return output


@quant_register(OpType.Cosh)
def cosh_quantize(self, *args):
    self.attrs['lambda_func'] = torch.cosh
    self.attrs['out_signed'] = False or self.force_dtype_int
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')


@op_register(OpType.Cosh)
def cosh(self, *args):
    def approximated_float_forward(self,  inp_tensor):
        if self.approximated and "lut" in self.constants:
            lut = self.constants["lut"].betensor
            sinh_out = lookup_float_index_lut(
                inp_tensor*0.5, lut, self.params['index_scale_value'], self.params['index_offset_value'], mirror_mode=True, value_offset_for_mirror_mode=self.params['value_offset_value'])
            out = 2 * sinh_out * sinh_out + 1
        else:
            out = cosh_func(inp_tensor)
        return out
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self,  x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@approx_register(OpType.Cosh)
def cosh_approx(self, *args):
    def set_min_max(inp, use_dynamic_lut):
        if not use_dynamic_lut:
            clip_min = 0
            clip_max = 10
        else:
            clip_min = 0
            clip_max = max(abs(inp.min)/2, abs(inp.max)/2)
        return clip_min, clip_max

    self.attrs['set_min_max'] = set_min_max
    self.attrs['lambda_func'] = cosh_func
    self.attrs['out_signed'] = False
    activation_module.unknown_approx(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('set_min_max')
    self.attrs.pop('out_signed')
    if self.get_param('is_perf_mode', optional=True, default_value=False) and 'lut' in self.constants:
        self.params['lut_mode'] = 'MIRROR'
