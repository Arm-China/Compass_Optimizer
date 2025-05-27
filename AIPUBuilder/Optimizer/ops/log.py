# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
import torch


@quant_register(OpType.Log)
def log_quantize(self, *args):
    self.attrs['lambda_func'] = torch.log
    self.attrs['out_signed'] = True
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')


@op_register(OpType.Log)
def log(self, *args):
    def approximated_float_forward(self,  inp_tensor):
        if self.approximated and "lut" in self.constants:
            lut = self.constants["lut"].betensor
            out = lookup_float_index_lut(
                inp_tensor, lut, self.params['index_scale_value'], self.params['index_offset_value'], mirror_mode=False, value_offset_for_mirror_mode=self.params['value_offset_value'])
        else:
            out = torch.log(inp_tensor)
        return out
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self,  x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@approx_register(OpType.Log)
def log_approx(self, *args):
    def set_min_max(inp, use_dynamic_lut):
        if use_dynamic_lut:
            clip_min = inp.min
            clip_max = inp.max
        else:
            clip_min = 1e-5
            clip_max = 256
        return clip_min, clip_max

    self.attrs['set_min_max'] = set_min_max
    self.attrs['lambda_func'] = torch.log
    self.attrs['out_signed'] = False
    activation_module.unknown_approx(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('set_min_max')
    self.attrs.pop('out_signed')
    if "lut" in self.constants:
        lut = self.constants["lut"].betensor
        inf_mask = float('-inf') == lut
        if True in inf_mask:
            valid_value = torch.log(torch.tensor(OPT_EPSILON, device=self.inputs[0].betensor.device))
            lut[inf_mask] = valid_value
            self.constants["lut"].betensor = lut
