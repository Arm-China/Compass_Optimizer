# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import math
import torch.nn as nn


@op_register(OpType.MaxPoolingWithArgMax)
def maxpoolingwithArgmax(self, *args):
    inp = self.inputs[0].betensor.float()
    n, h, w, c = inp.shape
    out_n = self.current_batch_size
    _, out_h, out_w, out_c = self.outputs[0].ir_shape
    kh, kw = (self.get_param('kernel_y'), self.get_param('kernel_x'))
    sh, sw = (self.get_param('stride_y'), self.get_param('stride_x'))
    pt, pl = (self.get_param('pad_top'), self.get_param('pad_left'))
    pb, pr = (self.get_param('pad_bottom'), self.get_param('pad_right'))
    dh, dw = (self.get_param('dilation_y'), self.get_param('dilation_x'))
    flatten_dim = self.get_param('flatten_dim')
    ceil_mode = self.get_param('ceil_mode')
    # row-major when storage_order = 0; column-major when storage_order = 1
    storage_order = int(self.get_param('storage_order'))
    padding = (pl, pr, pt, pb)
    input_data = nhwc2nchw(inp)
    input_data = nn.functional.pad(input_data, padding, value=-2.**31)

    #default is flatten_dim(HW)
    func = torch.nn.MaxPool2d(kernel_size=(kh, kw), stride=(sh, sw), padding=0,
                              dilation=(dh, dw), return_indices=True, ceil_mode=ceil_mode)
    value, indics = func(input_data)

    invalid_pad_mask = value == -2.**31
    current_outh, current_outw = value.shape[2:4]
    pad_h = h + pt + pb
    pad_w = w + pl + pr
    value[invalid_pad_mask] = 0
    indics[invalid_pad_mask] = -(pad_h*pad_w*c*n+2)
    if current_outh < out_h or current_outw < out_w:
        auto_pad_h = out_h - current_outh
        auto_pad_w = out_w - current_outw
        padding = (0, auto_pad_w, 0, auto_pad_h)  # padding_left,padding_right,padding_top,padding_bottom)
        pad_val = -self.inputs[0].zerop[0]
        value = torch.nn.functional.pad(value, padding, value=pad_val)
        indics = torch.nn.functional.pad(indics, padding, value=-(pad_h*pad_w*c*n+2)).long()
    H = indics // pad_w
    W = indics % pad_w
    H = H - pt
    W = W - pl
    H = nchw2nhwc(H)
    W = nchw2nhwc(W)
    indics = H * w + W
    if storage_order:
        # invalid indics will make H negative
        indics[H >= 0] = h * W[H >= 0] + H[H >= 0]
    value = nchw2nhwc(value)

    if flatten_dim == 'HW':
        pass
    elif flatten_dim == 'HWC':
        for channel in range(indics.shape[3]):
            indics[:, :, :, channel] = indics[:, :, :, channel] * c + channel
    elif flatten_dim == 'NHWC':
        for batch in range(indics.shape[0]):
            for channel in range(indics.shape[3]):
                indics[batch, :, :, channel] = batch * h * w * c + indics[batch, :, :, channel] * c + channel
    elif flatten_dim == 'NCHW':
        for batch in range(indics.shape[0]):
            for channel in range(indics.shape[3]):
                indics[batch, :, :, channel] = h * w * c * batch + h * w * channel + indics[batch, :, :, channel]
    else:
        OPT_FATAL("unsupported method: %s for MaxPoolingWithArgMax(type) in node:%s" % (flatten_dim, self.name))
    indics[indics < 0] = 0
    self.outputs[0].betensor = value
    self.outputs[1].betensor = indics

    return value, indics


@quant_register(OpType.MaxPoolingWithArgMax)
def maxpoolingwithArgmax_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    # for indice no quantize
    q_bits_activation = self.attrs["q_bits_activation"]
    out = self.outputs[1]
    out.scale = 1.0
    out.zerop = 0
    out.qbits = max(32, q_bits_activation)
    out.dtype = bits2dtype(out.qbits, is_signed=True or self.force_dtype_int)
    out.qinvariant = True
