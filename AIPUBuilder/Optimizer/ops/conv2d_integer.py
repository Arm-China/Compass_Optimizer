# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.conv import *
import torch

###########################################################
# layer_id=2
# layer_name=ConvInteger_0
# layer_type=ConvInteger
# layer_bottom=[Placeholder_0_post_transpose]
# layer_bottom_shape=[[2,51,52,6]]
# layer_bottom_type=[uint8]
# layer_top=[ConvInteger_0]
# layer_top_shape=[[2,45,45,10]]
# layer_top_type=[int32]
# weights_type=uint8
# weights_offset=0
# weights_size=3360
# weights_shape=[10,6,7,8]
# num_output=10
# kernel_x=8
# kernel_y=7
# stride_x=1
# stride_y=1
# pad_left=0
# pad_right=0
# pad_top=0
# pad_bottom=0
# dilation_x=1
# dilation_y=1
# group=1
# x_zero_point=23
# w_zero_point=56
###########################################################


@quant_register(OpType.ConvInteger)
def convinteger_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    out = self.outputs[0]

    x_zero_point = self.get_param('x_zero_point')
    w_zero_point = self.get_param('w_zero_point')

    # set weights quantization
    weights = self.constants['weights']
    weights_dtype = weights.ir_dtype
    weights.scale = 1
    weights.zerop = -w_zero_point
    weights.qmin, weights.qmax = dtype2range(weights_dtype)
    weights.qbits = dtype2bits(weights_dtype)
    weights.dtype = weights_dtype
    weights.qinvariant = True
    weights.betensor = linear_quantize_clip(weights.betensor, weights.scale, 0, weights.qmin, weights.qmax)

    # set bias quantization
    w = weights.betensor
    bias_t = self.constants['biases']
    bias = bias_t.betensor
    bias_qbits = 32
    bias = bias - (w - w_zero_point).reshape(w.shape[0], -1).sum(dim=1) * x_zero_point
    bias_t.scale = 1
    bias_t.zerop = 0
    bias_t.qmin = -2**(bias_qbits-1)
    bias_t.qmax = 2**(bias_qbits-1) - 1
    bias_t.betensor = torch.clamp(bias, bias_t.qmin, bias_t.qmax)
    bias_t.qbits = bias_qbits
    bias_t.dtype = bits2dtype(bias_qbits, is_signed=True)
    bias_t.qinvariant = True

    # set output quantization
    out.scale = 1
    out.zerop = 0
    out.qbits = 32
    out.qmin, out.qmax = bits2range(out.qbits, True)
    out.dtype = bits2dtype(out.qbits, True)
    out.qinvariant = True

    inp.scale = 1.0
    inp.zerop = -x_zero_point
    inp.qinvariant = True


@op_register(OpType.ConvInteger)
def convinteger(self, *args):

    if not self.quantized:
        x_zero_point = self.get_param('x_zero_point')
        w_zero_point = self.get_param('w_zero_point')
        input_bak = self.inputs[0].betensor.clone()
        weights_bak = self.constants['weights'].betensor.clone()

        self.inputs[0].betensor -= x_zero_point
        self.constants['weights'].betensor -= w_zero_point

        output = conv2d(self, *args)

        self.inputs[0].betensor = input_bak
        self.constants['weights'].betensor = weights_bak
    else:
        output = conv2d(self, *args)

    self.outputs[0].betensor = output
    return output
