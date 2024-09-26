# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.conv import *
from AIPUBuilder.Optimizer.ops.activation import apply_with_activation
import torch


@op_register(OpType.Convolution3D)
def conv3d(self, *args):
    inp = self.inputs[0].betensor.float()
    weights = self.constants["weights"].betensor.clone().float()
    bias = self.constants['biases'].betensor.clone().float()
    pad_val = 0
    aasrb = self.get_param('remain_shift',
                           optional=True, default_value=None)
    if self.quantized:
        # input's zerop has been absorbed to bias.
        # inp += self.inputs[0].zerop
        pad_val = -self.inputs[0].zerop[0]
        weights += self.constants["weights"].broadcast_zerop
        bias += self.constants['biases'].broadcast_zerop

    inp = inp.permute(0, 4, 1, 2, 3)
    weights = weights.permute(0, 4, 3, 1, 2)  # [out_c, h, w, d, in_c] -> [out_c, in_c, d, h, w]

    # kernel_size = self.get_param('kernel_z'), self.get_param('kernel_y'), self.get_param('kernel_x')
    stride = self.get_param('stride_z'), self.get_param('stride_y'), self.get_param('stride_x')
    padding = (self.get_param('pad_x_begin'), self.get_param('pad_x_end'),
               self.get_param('pad_y_begin'), self.get_param('pad_y_end'),
               self.get_param('pad_z_begin'), self.get_param('pad_z_end'))
    dilation = (self.get_param('dilation_z'), self.get_param('dilation_y'), self.get_param('dilation_x'))
    inp = torch.nn.functional.pad(inp, padding, value=pad_val)
    x = torch.nn.functional.conv3d(inp,
                                   weights,
                                   bias if aasrb is None else None,
                                   stride=stride,
                                   padding=0,
                                   dilation=dilation,
                                   groups=self.get_param("group")
                                   )
    x = x.permute(0, 2, 3, 4, 1)
    shift_bk = None
    if self.quantized and aasrb is not None and (dtype2bits(self.constants["weights"].dtype) > 8 or dtype2bits(self.inputs[0].dtype) > 8):
        self.outputs[0].betensor = apply_with_activation(self, x,
                                                         *args, aasrb=(aasrb, bias))
        return self.outputs[0].betensor
    self.outputs[0].betensor = apply_with_activation(self, x, *args)
    return self.outputs[0].betensor


@quant_register(OpType.Convolution3D)
def conv3d_quantize(self, *args):
    linear_op_quantize(self, *args)
    absorb_input_zp_to_bias_and_compress_bias_for_aiff(self, *args)
