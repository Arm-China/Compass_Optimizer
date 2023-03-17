# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.ops.conv import *
from AIPUBuilder.Optimizer.ops.activation import apply_with_activation
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch


@quant_register(OpType.ConvTranspose3D)
def conv_transpose3d_quantize(self, *args):
    group = self.get_param('group')
    w = self.constants["weights"]
    w.betensor = w.betensor.repeat(group, 1, 1, 1, 1)
    if group > 1:
        w.min_key_axis = w.min_key_axis.repeat(group)
        w.max_key_axis = w.max_key_axis.repeat(group)
    linear_op_quantize(self, *args)
    absorb_input_zp_to_bias(self, *args)
    w.betensor = w.betensor[:self.outputs[0].shape[-1]//group]
    ls = w.scale
    w.scale = ls[:self.outputs[0].shape[-1]//group] if isinstance(ls, torch.Tensor) else ls
    lz = w.zerop
    w.zerop = lz[:self.outputs[0].shape[-1]//group] if isinstance(lz, torch.Tensor) else lz
    clear_lower_bits_for_bias(self, *args)


@op_register(OpType.ConvTranspose3D)
def conv_transpose3d(self, *args):
    inp = self.inputs[0].betensor.float()
    weights = self.constants["weights"].betensor.clone().float()
    # weights = weights.permute(4, 0, 3, 1, 2)  # [out_c, h, w, d, in_c] -> [in_c, out_c, d, h, w]
    bias = self.constants['biases'].betensor.clone().float()
    if self.quantized:
        inp += self.inputs[0].zerop
        w_zp = self.constants["weights"].zerop
        w_zshape = [1] * weights.dim()
        w_zshape[0] = -1
        weights += w_zp.reshape(w_zshape) if isinstance(w_zp, torch.Tensor) else w_zp
        # pass inputs'zp as padding value to torch.convtranspose is inconvenient
        # so bias will release inputs'zp out of it first, and inp should add its zp firstly.
        bias -= compute_input_zp_mul_reduce_weight(self.inputs[0].zerop, weights).repeat(self.get_param('group'))
        bias += self.constants['biases'].zerop

    inp = inp.permute(0, 4, 1, 2, 3)
    weights = weights.permute(4, 0, 3, 1, 2)  # [out_c, h, w, d, in_c] -> [in_c, out_c, d, h, w]
    stride = self.get_param('stride_z'), self.get_param('stride_y'), self.get_param('stride_x')
    dilation = (self.get_param('dilation_z'), self.get_param('dilation_y'), self.get_param('dilation_x'))

    # new_padding = k-param's pad -1
    # padding is for conv_transpose2d inter's fill,not for orignal inp
    # step1:padding rules are fill zero in feature every stride,such feature H is 8,need 7 zeros are filled
    # step2:conv_transpose2d inter new_padding = k-param's pad -1
    # step3: new input size is h+(h-1)*(stride-1)+2*new_padding
    # step4: stride=1
    # step5:conv2d, outputsize = h+(h-1)*(stride-1)+2*new_padding-k+1

    x = torch.nn.functional.conv_transpose3d(inp,
                                             weights,
                                             bias,
                                             stride=stride,
                                             padding=0,
                                             output_padding=0,
                                             groups=self.get_param("group"),
                                             dilation=dilation
                                             )
    # now output size is h+(h-1)*(stride-1)+(k-1)+(k-1), but
    # expected outsize =h+(h-1)*(stride-1)+(k-pad_top-1)+(k-pad_bottom-1),so need crop size
    # top_start=(k-1)-(k-pad_top-1)=pad_top
    # bottom_end = output size-((k-1)-(k-pad_bottom-1))=h-pad_bottom
    d, h, w = x.shape[-3:]
    pad_x_begin, pad_x_end = self.get_param('pad_x_begin'), self.get_param('pad_x_end')
    pad_y_begin, pad_y_end = self.get_param('pad_y_begin'), self.get_param('pad_y_end')
    pad_z_begin, pad_z_end = self.get_param('pad_z_begin'), self.get_param('pad_z_end')
    # if pad_z_begin > pad_z_end:
    #     pad_z_begin = pad_z_begin - 1
    #     pad_z_end = pad_z_end + 1
    #
    # if pad_y_begin > pad_y_end:
    #     pad_y_begin = pad_y_begin - 1
    #     pad_y_end = pad_y_end + 1
    #
    # if pad_x_begin > pad_x_end:
    #     pad_x_begin = pad_x_begin - 1
    #     pad_x_end = pad_x_end + 1

    crop_x = x[..., pad_z_begin:d-pad_z_end, pad_y_begin:h-pad_y_end, pad_x_begin:w-pad_x_end]
    x = crop_x.permute(0, 2, 3, 4, 1)

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
