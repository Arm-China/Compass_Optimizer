# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    if self.quantized:
        # input's zerop has been absorbed to bias.
        # inp += self.inputs[0].zerop
        pad_val = -self.inputs[0].zerop
        w_zp = self.constants["weights"].zerop
        w_zshape = [1] * weights.dim()
        w_zshape[0] = -1
        weights += w_zp.reshape(w_zshape) if isinstance(w_zp, torch.Tensor) else w_zp
        bias += self.constants['biases'].zerop

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
                                   bias,
                                   stride=stride,
                                   padding=0,
                                   dilation=dilation,
                                   groups=self.get_param("group")
                                   )
    x = x.permute(0, 2, 3, 4, 1)

    requant_scale = 1
    requant_shift = 0
    if self.quantized:
        if 'scale_value' in self.params:
            requant_scale = self.params['scale_value']
        elif "scale" in self.constants:
            requant_scale = self.constants["scale"].betensor

        if 'shift_value' in self.params:
            requant_shift = self.params['shift_value']
        elif "shift" in self.constants:
            requant_shift = self.constants["shift"].betensor

    x = apply_with_activation(self, x,
                              self.inputs[0].scale * self.constants["weights"].scale, 0,
                              requant_scale,
                              requant_shift,
                              *args)

    self.outputs[0].betensor = x
    return x


@quant_register(OpType.Convolution3D)
def conv3d_quantize(self, *args):
    linear_op_quantize(self, *args)
    absorb_input_zp_to_bias_and_compress_bias_for_aiff(self, *args)
