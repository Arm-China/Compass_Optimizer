# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

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
    absorb_input_zp_to_bias_and_compress_bias_for_aiff(self, *args)

    w.betensor = w.betensor[:self.outputs[0].ir_shape[-1]//group]
    ls = w.scale
    w.scale = ls[:self.outputs[0].ir_shape[-1]//group] if isinstance(ls, torch.Tensor) else ls
    lz = w.zerop
    w.zerop = lz[:self.outputs[0].ir_shape[-1]//group] if isinstance(lz, torch.Tensor) else lz


@op_register(OpType.ConvTranspose3D)
def conv_transpose3d(self, *args):
    inp = self.inputs[0].betensor.float()
    weights = self.constants["weights"].betensor.clone().float()
    # weights = weights.permute(4, 0, 3, 1, 2)  # [out_c, h, w, d, in_c] -> [in_c, out_c, d, h, w]
    bias = self.constants['biases'].betensor.clone().float()
    aasrb = self.get_param('remain_shift',
                           optional=True, default_value=None)
    if self.quantized:
        inp += self.inputs[0].broadcast_zerop
        weights += self.constants['weights'].broadcast_zerop
        # pass inputs'zp as padding value to torch.convtranspose is inconvenient
        # so bias will release inputs'zp out of it first, and inp should add its zp firstly.
        bias -= compute_input_zp_mul_reduce_weight(self.inputs[0].zerop, weights).repeat(self.get_param('group'))
        bias += self.constants['biases'].broadcast_zerop

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
                                             bias if aasrb is None else None,
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
    if self.quantized and aasrb is not None and (dtype2bits(self.constants["weights"].dtype) > 8 or dtype2bits(self.inputs[0].dtype) > 8):
        self.outputs[0].betensor = apply_with_activation(self, x,
                                                         *args, aasrb=(aasrb, bias))
        return self.outputs[0].betensor
    self.outputs[0].betensor = apply_with_activation(self, x, *args)
    return self.outputs[0].betensor
