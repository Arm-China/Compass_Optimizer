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
    inp = self.inputs[0]
    dev = inp.betensor.device
    approx_params = self.get_attrs('approx_params', optional=True, default_value=[0])
    method = int(approx_params[0] if len(approx_params) > 0 else 0)
    min_compatible_zhouyi_target = self.attrs["min_compatible_zhouyi_target"].upper()
    lut_items_in_bits = Target.aiff_lut_items_in_bits(min_compatible_zhouyi_target)
    if 1 == method and Target.optimized_target_level(min_compatible_zhouyi_target) >= 2:
        q_mode_activation = self.attrs["q_mode_activation"]
        use_dynamic_lut = len(approx_params) > 1 and approx_params[1] > 0
        bak_min = inp.min
        bak_max = inp.max
        if not use_dynamic_lut:
            inp.min = 0
            inp.max = 6
        inp.min = 0
        inp.max = max(abs(inp.min), abs(inp.max))
        index_scale, index_offset, _, _, _ = get_linear_quant_params_from_tensor(
            inp, QuantMode.to_asymmetric(QuantMode.to_per_tensor(q_mode_activation)), lut_items_in_bits, False)
        inp.min = bak_min
        inp.max = bak_max
        lut = linear_dequantize(torch.arange(0, 2**lut_items_in_bits, device=dev), index_scale, index_offset)
        value_offset = -0.5
        lut = phi_tanh_approx(lut) + value_offset
        lut = to_fp24(lut)
        self.constants["lut"] = PyTensor(self.name + "/plh_lut", lut.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
        self.params['is_perf_mode'] = True
        self.params['lut_mode'] = 'MIRROR'
        self.params['index_scale_value'] = index_scale
        self.params['index_scale_type'] = Dtype.FP32
        self.params['index_offset_value'] = index_offset
        self.params['index_offset_type'] = Dtype.FP32
        self.params['value_offset_value'] = value_offset
        self.params['value_offset_type'] = Dtype.FP32
    else:
        # not suit for aiff, need use tpc to implement a high accuracy version
        self.params['is_perf_mode'] = False


@op_register(OpType.GELU)
def gelu(self, *args):
    apprx = self.get_param('approximate').lower()  # NONE, Tanh
    if apprx not in ['none', 'tanh']:
        OPT_ERROR('GELU dont support approximation method:%s' % apprx)
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self, x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')

    return self.outputs[0].betensor
