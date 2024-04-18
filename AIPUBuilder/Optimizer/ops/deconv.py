# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.ops.conv import *
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.activation import apply_with_activation
import torch.nn as nn
import torch


@quant_register(OpType.ConvTranspose)
def deconv2d_quantize(self, *args):
    group = self.get_param('group')
    w = self.constants["weights"]
    w.betensor = w.betensor.repeat(group, 1, 1, 1)
    if group > 1:
        w.min_key_axis = w.min_key_axis.repeat(group)
        w.max_key_axis = w.max_key_axis.repeat(group)
    linear_op_quantize(self, *args)
    absorb_input_zp_to_bias_and_compress_bias_for_aiff(self, *args)
    w.betensor = self.constants["weights"].betensor[:self.outputs[0].ir_shape[-1]//group]
    ls = w.scale
    w.scale = ls[:self.outputs[0].ir_shape[-1]//group] if isinstance(ls, torch.Tensor) else ls
    lz = w.zerop
    w.zerop = lz[:self.outputs[0].ir_shape[-1]//group] if isinstance(lz, torch.Tensor) else lz


@op_register(OpType.ConvTranspose)
def deconv2d(self, *args):
    inp = self.inputs[0].betensor.float()
    weights = self.constants["weights"].betensor.clone().float()
    bias = self.constants['biases'].betensor.clone().float()
    if self.quantized:
        inp += self.inputs[0].zerop
        w_zp = self.constants["weights"].broadcast_zerop
        weights += w_zp
        # pass inputs'zp as padding value to torch.convtranspose is inconvenient
        # so bias will release inputs'zp out of it first, and inp should add its zp firstly.
        bias -= compute_input_zp_mul_reduce_weight(self.inputs[0].zerop, weights).repeat(self.get_param('group'))
        bias += self.constants['biases'].zerop

    inp = nhwc2nchw(inp)
    weights = nhwc2nchw(weights)
    weights = weights.permute(1, 0, 2, 3)
    dilation_x = self.get_param("dilation_x")
    dilation_y = self.get_param("dilation_y")
    group = self.get_param('group')
    # new_padding = k-param's pad -1
    # padding is for conv_transpose2d inter's fill,not for orignal inp
    # step1:padding rules are fill zero in feature every stride,such feature H is 8,need 7 zeros are filled
    # step2:conv_transpose2d inter new_padding = k-param's pad -1
    # step3: new input size is h+(h-1)*(stride-1)+2*new_padding
    # step4: stride=1
    # step5:conv2d, outputsize = h+(h-1)*(stride-1)+2*new_padding-k+1

    bottom_end = self.get_param('pad_bottom')
    top_start = self.get_param('pad_top')

    right_end = self.get_param('pad_right')
    left_start = self.get_param('pad_left')
    stride_x = self.get_param("stride_x")
    stride_y = self.get_param("stride_y")
    kernel_x = self.get_param("kernel_x")
    kernel_y = self.get_param("kernel_y")
    outpadding_x = self.get_param("output_padding_x", optional=True, default_value=0)
    outpadding_y = self.get_param("output_padding_y", optional=True, default_value=0)
    # padH_out = 0
    # padW_out = 0
    # Woutpad = 0
    # Houtpad = 0

    # total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
    # If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
    # Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

    # if (kernel_x-stride_x)//2>=0 :
    #     if (right_end+left_start)<=(kernel_x-stride_x) and (kernel_x-stride_x)%2!=0:
    #         padW_out = max(left_start-(kernel_x-stride_x)//2,0)
    # else:
    #     padW_out = (kernel_x-stride_x)//2+1
    #     Woutpad = -(right_end+padW_out) if right_end+padW_out<0 else 0
    # if (kernel_y-stride_y)//2>=0:
    #     if (top_start+bottom_end)<=(kernel_y-stride_y) and (kernel_y-stride_y)%2!=0:
    #         padH_out = max(top_start-(kernel_y-stride_y)//2,0)
    # else:
    #     padH_out = (kernel_y-stride_y)//2+1
    #     Houtpad = -(bottom_end+padH_out) if bottom_end+padH_out<0 else 0

    if outpadding_y != (self.outputs[0].ir_shape[1]-dilation_y*(kernel_y-1)-1+bottom_end+top_start) % stride_y:
        OPT_DEBUG('output shape y may be not correct,please check outpadding_y')
    if outpadding_x != (self.outputs[0].ir_shape[2]-dilation_x*(kernel_x-1)-1+right_end+left_start) % stride_x:
        OPT_DEBUG('output shape x may be not correct,please check outpadding_x')
    x = nn.functional.conv_transpose2d(inp,
                                       weights,
                                       bias,
                                       stride=(stride_y, stride_x),
                                       padding=(0, 0),
                                       output_padding=(outpadding_y, outpadding_x),
                                       groups=group,
                                       dilation=(dilation_y, dilation_x)
                                       )
    # now output size is h+(h-1)*(stride-1)+(k-1)+(k-1), but
    # expected outsize =h+(h-1)*(stride-1)+(k-pad_top-1)+(k-pad_bottom-1),so need crop size
    # top_start=(k-1)-(k-pad_top-1)=pad_top
    # bottom_end = output size-((k-1)-(k-pad_bottom-1))=h-pad_bottom
    h = x.shape[-2]
    w = x.shape[-1]
    crop_x = x[..., top_start:h-bottom_end, left_start:w-right_end]
    x = nchw2nhwc(crop_x)

    self.outputs[0].betensor = apply_with_activation(self, x, *args)
    return self.outputs[0].betensor
