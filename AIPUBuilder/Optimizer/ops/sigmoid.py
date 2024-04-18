# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import AIPUBuilder.Optimizer.ops.activation as activation_module
from AIPUBuilder.Optimizer.utils.math_utils import *
import torch

register_optype('Sigmoid')


@op_register(OpType.Sigmoid)
def sigmoid_forward(self, *args):
    def approximated_float_forward(self,  inp_tensor):
        if self.approximated and "lut" in self.constants:
            lut = self.constants["lut"].betensor
            out = lookup_float_index_lut(inp_tensor, lut,
                                         self.params['index_scale_value'],
                                         self.params['index_offset_value'],
                                         mirror_mode=True,
                                         value_offset_for_mirror_mode=self.params['value_offset_value'])
        else:
            out = torch.sigmoid(inp_tensor)
        return out
    self.attrs['lambda_func'] = lambda x: approximated_float_forward(self,  x)
    self.outputs[0].betensor = activation_module.unknown_activation(self, *args)
    self.attrs.pop('lambda_func')
    return self.outputs[0].betensor


@quant_register(OpType.Sigmoid)
def sigmoid_quantize(self, *args):
    def sigmoid_func(x): return torch.sigmoid(x)
    self.attrs['lambda_func'] = sigmoid_func
    self.attrs['out_signed'] = False or self.force_dtype_int
    offset = 0.0
    if self.type in [OpType.BasicLSTM, OpType.GRUv3, OpType.GRUv1]:
        offset = torch.sigmoid(torch.tensor(0.0)).item()
    self.attrs['mirror_offset'] = offset

    activation_module.unknown_quantize(self, *args)

    for k in ['lambda_func', 'out_signed', 'mirror_offset']:
        self.attrs.pop(k)


@approx_register(OpType.Sigmoid)
def sigmoid_approx(self, *args):
    inp = self.inputs[0]
    dev = inp.device
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
        lut = torch.nn.functional.sigmoid(lut) + value_offset
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
