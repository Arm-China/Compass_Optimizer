# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.bn import *
import torch

register_optype('BiasAdd')


@quant_register(OpType.BiasAdd)
def bias_add_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    if inp.key_axis is not None or out.key_axis is not None:
        quant_type = self.attrs.get('quant_type')
        fpx_quantize = True if quant_type != 'disable' and QuantType.is_float(quant_type) else False
        q_mode_activation = self.attrs["q_mode_activation"]
        q_bits_activation = self.attrs["q_bits_activation"]
        out.qinvariant = False
        out.qbits = q_bits_activation
        if fpx_quantize:
            out.dtype = QuantType._to_Dtype(QuantType.activation_type(quant_type))
            out.scale, out.zerop, out.qmin, out.qmax = get_fpx_quant_params_from_tensor(
                out, q_mode_activation, out.dtype)
        else:
            out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
                out, q_mode_activation, out.qbits, is_signed=True)
        b = self.constants['biases']
        b.scale = 1.0
        b.zerop = 0
        b.dtype = Dtype.FP32
        b.qbits = 32
        b.qinvariant = False
        self.constants['scale1'] = PyTensor('scale1', 1.0 / inp.broadcast_scale, dtype=Dtype.FP32)
        self.constants['scale2'] = PyTensor('scale2', out.broadcast_scale, dtype=Dtype.FP32)
    else:
        # bias_add is equal to batchnorm with weights == 1
        self.attrs["q_mode_weight"] = self.attrs["q_mode_activation"]
        self.attrs["q_mode_bias"] = self.attrs["q_mode_weight"]
        self.constants["weights"] = self.constants["weights_bk"]
        batch_norm_quantize(self, *args)
        self.constants.pop('weights')
    self.constants.pop('weights_bk')


@op_register(OpType.BiasAdd)
def bias_add_forward(self, *args):
    if not self.quantized:
        if 'weights_bk' not in self.constants.keys():
            self.constants["weights_bk"] = PyTensor(self.name + '/temp_weights_bk')
            self.constants["weights_bk"].betensor = torch.ones_like(self.constants["biases"].betensor)
            self.constants['weights_bk'].ir_shape = self.constants["biases"].shape
            self.constants['weights_bk'].ir_dtype = self.constants["biases"].ir_dtype
    self.constants["weights"] = PyTensor(self.name + '/temp_weights')
    self.constants["weights"].betensor = torch.ones_like(self.constants["biases"].betensor)
    self.constants['weights'].ir_shape = self.constants["biases"].shape
    self.constants['weights'].ir_dtype = self.constants["biases"].ir_dtype
    aflag = False
    if 'axis' not in self.params:
        aflag = True
        input_dim = self.inputs[0].betensor.dim()
        self.params['axis'] = input_dim - 1
    if all(sname in self.constants.keys() for sname in ['scale1', 'scale2', ]):
        # act per channel quantization
        scale1 = self.constants['scale1'].betensor
        scale2 = self.constants['scale2'].betensor
        z = self.inputs[0].betensor * scale1 + self.constants['biases'].betensor
        z = linear_quantize_clip(z.float(
        ), scale2, 0, self.outputs[0].qmin, self.outputs[0].qmax, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', self.outputs[0].dtype))
        self.outputs[0].betensor = z
    else:
        batch_norm(self, *args)
    if aflag:
        self.params.pop('axis')
    self.constants.pop('weights')
    return self.outputs[0].betensor
