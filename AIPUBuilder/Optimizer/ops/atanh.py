# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
from AIPUBuilder.Optimizer.utils.math_utils import *
import torch

register_optype('Atanh')


@op_register(OpType.Atanh)
def atanh_forward(self, *args):
    def approximated_float_forward(self,  inp_tensor):
        if self.approximated and "lut" in self.constants:
            lut = self.constants["lut"].betensor
            out = lookup_float_index_lut(inp_tensor, lut,
                                         self.params['index_scale_value'],
                                         self.params['index_offset_value'],
                                         mirror_mode=True,
                                         value_offset_for_mirror_mode=self.params['value_offset_value'])
        else:
            out = torch.atanh(inp_tensor)
            nan_mask = torch.isnan(out)
            infi_mask = torch.isinf(out)
            if True in nan_mask:
                out[nan_mask] = OPT_INT_MAX
                OPT_WARN(
                    f"{self} there are nan value in the output, please check the input range.Currently,nan is overridden with a maximum value {OPT_INT_MAX}, but this may make the result abnormal!")

            if True in infi_mask:
                OPT_WARN(
                    f"{self} the output exists +/-INF, for the convenience of subsequent quantification, we do +/-INF clamp to [{OPT_INT_MIN},{OPT_INT_MAX}].")
                out = torch.clamp(out, OPT_INT_MIN, OPT_INT_MAX)
        return out
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self,  x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@quant_register(OpType.Atanh)
def atanh_quantize(self, *args):
    self.attrs['lambda_func'] = torch.atanh
    self.attrs['out_signed'] = True
    activation_module.unknown_quantize(self, *args)
    for k in ['lambda_func', 'out_signed']:
        self.attrs.pop(k)


@approx_register(OpType.Atanh)
def atanh_approx(self, *args):
    def set_min_max(inp, use_dynamic_lut):
        if not use_dynamic_lut:
            clip_min = 0
            clip_max = 1 - torch.finfo(torch.float32).eps
        else:
            clip_min = 0
            clip_max = max(abs(inp.min), abs(inp.max))
        return clip_min, clip_max

    self.attrs['set_min_max'] = set_min_max
    self.attrs['lambda_func'] = torch.atanh
    self.attrs['out_signed'] = False
    activation_module.unknown_approx(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('set_min_max')
    self.attrs.pop('out_signed')

    self.params['lut_mode'] = 'MIRROR'
