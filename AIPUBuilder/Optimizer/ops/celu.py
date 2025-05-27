# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
import torch

#celu:  max(0,x) + min(0,alpha*(exp(x/alpha)-1))

register_optype('CELU')


@quant_register(OpType.CELU)
def celu_quantize(self, *args):

    def celu_lambda(x): return torch.nn.functional.celu(x, float(self.get_param("alpha")))
    self.attrs['lambda_func'] = celu_lambda
    self.attrs['out_signed'] = True
    activation_module.unknown_quantize(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('out_signed')

    self.params.pop('alpha')


@op_register(OpType.CELU)
def celu(self, *args):
    def approximated_float_forward(self,  inp_tensor):
        if self.approximated and "lut" in self.constants:
            lut = self.constants["lut"].betensor
            out = lookup_float_index_lut(
                inp_tensor, lut, self.params['index_scale_value'], self.params['index_offset_value'], mirror_mode=False, value_offset_for_mirror_mode=self.params['value_offset_value'])
        else:
            alpha = float(self.get_param("alpha"))
            out = torch.nn.functional.celu(inp_tensor, alpha)
        return out
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self,  x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@approx_register(OpType.CELU)
def celu_approx(self, *args):
    def set_min_max(inp, use_dynamic_lut, alpha):
        import math
        negative_limit = math.log(1e-5) * alpha - 2
        # The value that crosses the boundary can be calculated based on the slope
        return negative_limit, 4

    self.attrs['set_min_max'] = lambda inp, use_dynamic_lut: set_min_max(
        inp, use_dynamic_lut, float(self.get_param("alpha")))
    self.attrs['lambda_func'] = lambda x: torch.nn.functional.celu(x, float(self.get_param("alpha")))
    self.attrs['out_signed'] = False
    activation_module.unknown_approx(self, *args)
    self.attrs.pop('lambda_func')
    self.attrs.pop('set_min_max')
    self.attrs.pop('out_signed')
