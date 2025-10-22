# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.ops.conv import *
from AIPUBuilder.Optimizer.ops.activation import apply_with_activation
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch.nn as nn


@op_register(OpType.FullyConnected)
def fc(self, *args):
    inp, bias, weights = None, None, None
    if self.constants['weights'].dtype in [Dtype.FP32, Dtype.FP16, Dtype.BFP16]:
        inp = self.inputs[0].betensor.float()
        bias = self.constants["biases"].betensor.float()
        weights = self.constants["weights"].betensor.float()
    else:
        inp = self.inputs[0].betensor.double()
        bias = self.constants["biases"].betensor.clone().double()
        weights = self.constants["weights"].betensor.clone().double()
    aasrb = self.get_param('remain_shift',
                           optional=True, default_value=None)

    if self.quantized:
        # input's zerop has been absorbed to bias.
        # inp += self.inputs[0].zerop
        bias += self.constants['biases'].broadcast_zerop
        if all(sname in self.constants.keys() for sname in ['scale0', 'scale1']):
            weights, scale1 = linear_dequantize_for_gptq_w4afp8(self.constants)
            x = (inp@weights.double().T)*scale1.T+bias
            self.outputs[0].betensor = apply_with_activation(self, x, *args)
            return self.outputs[0].betensor
        weights += self.constants["weights"].broadcast_zerop
        if aasrb is not None and (dtype2bits(self.constants["weights"].dtype) > 8 or dtype2bits(self.inputs[0].dtype) > 8):
            x = inp @ weights.T
            self.outputs[0].betensor = apply_with_activation(self, x,
                                                             *args, aasrb=(aasrb, bias))
            return self.outputs[0].betensor

        if all(sname in self.constants.keys() for sname in ['scale1', 'scale2', 'scale3']):
            # act per channel quantization
            a_scale = self.constants['scale1'].betensor
            b_scale = self.constants['scale2'].betensor
            c_scale = self.constants['scale3'].betensor
            x = inp @ weights.T
            y = x * a_scale * b_scale + bias
            self.quantized = False
            y = apply_with_activation(self, y, *args)
            self.quantized = True
            self.outputs[0].betensor = linear_quantize_clip(y.float(
            ), c_scale, 0, self.outputs[0].qmin, self.outputs[0].qmax, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', self.outputs[0].dtype))
            #torch.clamp((y * c_scale).round(), self.outputs[0].qmin, self.outputs[0].qmax)
            return self.outputs[0].betensor

    x = nn.functional.linear(inp, weights, bias,)
    self.outputs[0].betensor = apply_with_activation(self, x, *args)
    return self.outputs[0].betensor


@quant_register(OpType.FullyConnected)
def fc_quantize(self, *args):
    if all(ka is None for ka in [self.inputs[0].key_axis, self.outputs[0].key_axis]):
        conv2d_quantize(self, *args)
    else:
        # act per channel quantization
        q_mode_activation = self.attrs["q_mode_activation"]
        quant_type = self.attrs.get('quant_type')
        fpx_quantize = True if quant_type != 'disable' and QuantType.is_float(quant_type) else False
        q_mode_weight = self.attrs["q_mode_weight"]
        q_bits_weight = self.attrs["q_bits_weight"]
        q_bits_activation = self.attrs["q_bits_activation"]
        w = self.constants["weights"]
        out = self.outputs[0]
        out_signed = with_activation_out_is_signed(self) or self.force_dtype_int
        out.qbits = q_bits_activation
        if fpx_quantize:
            q_type_activation = QuantType.activation_type(quant_type)
            out.dtype = q_type_activation
            out.scale, out.zerop, out.qmin, out.qmax = get_fpx_quant_params_from_tensor(
                out, q_mode_activation, q_type_activation)
            q_type_weight = QuantType.weight_type(quant_type)
            w.dtype = q_type_weight
            w.qbits = dtype2bits(w.dtype)
            w.scale, w.zerop, w.qmin, w.qmax = get_fpx_quant_params_from_tensor(w, q_mode_weight, q_type_weight)
            w.betensor = linear_quantize_clip(w.betensor.float(), w.broadcast_scale, w.broadcast_zerop,
                                              w.qmin, w.qmax, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', w.dtype))
        else:
            out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
                out, q_mode_activation, q_bits_activation, is_signed=out_signed)
            w.scale, w.zerop, w.qmin, w.qmax, w.dtype = get_linear_quant_params_from_tensor(w, q_mode_weight, q_bits_weight,
                                                                                            is_signed=True)
            w.betensor = linear_quantize_clip(w.betensor, w.broadcast_scale, w.broadcast_zerop, w.qmin, w.qmax)
            w.qbits = q_bits_weight
        out.qinvariant = False
        w.qinvariant = False

        b = self.constants['biases']
        b.scale = 1.0
        b.zerop = 0
        b.dtype = Dtype.FP32
        b.qbits = 32
        b.qinvariant = False

        inp = self.inputs[0]

        self.constants['scale1'] = PyTensor('scale1', 1.0 / inp.broadcast_scale, dtype=Dtype.FP32)
        self.constants['scale2'] = PyTensor('scale2', 1.0 / w.broadcast_scale.T, dtype=Dtype.FP32)
        self.constants['scale3'] = PyTensor('scale3', out.broadcast_scale, dtype=Dtype.FP32)
