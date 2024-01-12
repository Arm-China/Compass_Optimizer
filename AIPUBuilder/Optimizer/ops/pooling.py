# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch.nn as nn
import torch
import math


@op_register(OpType.Pooling)
def pooling(self, *args):
    out = self.outputs[0]
    inp = self.inputs[0].betensor.float()
    count_include_pad = self.get_param(
        'count_include_pad', optional=True, default_value=True)
    kernel_size = self.get_param('kernel_y', optional=True, default_value=1), self.get_param("kernel_x")
    stride = self.get_param("stride_y", optional=True, default_value=1), self.get_param("stride_x")
    dilation = self.get_param("dilation_y", optional=True, default_value=1), self.get_param("dilation_x")
    padding = (self.get_param('pad_left'), self.get_param('pad_right'),
               self.get_param('pad_top', optional=True, default_value=0),
               self.get_param('pad_bottom', optional=True, default_value=0))

    if len(self.inputs[0].ir_shape) < 4:
        local_ceil_mode = False if math.floor(
            (self.inputs[0].ir_shape[1] - ((kernel_size[1] - 1) * dilation[1] + 1) + padding[0] + padding[1]) / stride[1]) + 1 == \
            out.ir_shape[1] else True
        inp = inp.unsqueeze(1)
    else:
        local_ceil_mode = False if math.floor(
            (self.inputs[0].ir_shape[1] - ((kernel_size[0] - 1) * dilation[0] + 1) + padding[2] + padding[3]) / stride[0]) + 1 == \
            out.ir_shape[1] and math.floor(
            (self.inputs[0].ir_shape[2] - ((kernel_size[1] - 1) * dilation[1] + 1) + padding[0] + padding[1]) / stride[1]) + 1 == \
            out.ir_shape[2] else True
    ceil_mode = local_ceil_mode

    # to calculate output shape and extra padding needed when ceil_mode=true
    oh = self.outputs[0].ir_shape[1] if len(self.inputs[0].ir_shape) == 4 else 1
    extra_ph = (oh - 1) * stride[0] + dilation[0] * (kernel_size[0] - 1) + \
        1 - inp.shape[1] - padding[2] - padding[3]
    ow = self.outputs[0].ir_shape[2] if len(self.inputs[0].ir_shape) == 4 else self.outputs[0].ir_shape[1]
    extra_pw = (ow - 1) * stride[1] + dilation[1] * (kernel_size[1] - 1) + \
        1 - inp.shape[2] - padding[0] - padding[1]

    if extra_pw + padding[1] >= kernel_size[1] and ceil_mode == True:
        OPT_WARN(
            'ceil_mode=True cause total padding width exceed kernel_width size(%d>=%d). Result maybe untrustworthy.' % (
                extra_pw + padding[1], kernel_size[1]))

    if extra_ph + padding[3] >= kernel_size[0] and ceil_mode == True:
        OPT_WARN(
            'ceil_mode=True cause total padding height exceed kernel_height size(%d>=%d). Result maybe untrustworthy.' % (
                extra_ph + padding[3], kernel_size[0]))

    inp = nhwc2nchw(inp)
    pmethod = self.get_param('method').upper()
    if pmethod == 'MAX':
        pvalue = torch.finfo(torch.float32).min if not self.quantized else self.inputs[0].qmin
        inp = torch.nn.functional.pad(inp, padding, value=pvalue)
        y = torch.nn.functional.max_pool2d(inp.float(),
                                           kernel_size=kernel_size,
                                           stride=stride, padding=0, dilation=dilation,
                                           ceil_mode=ceil_mode)
        out.betensor = nchw2nhwc(y)
    elif pmethod == 'AVG':
        pvalue = 0
        inp2 = torch.nn.functional.pad(
            (inp + self.inputs[0].zerop) if self.quantized else inp, padding, value=pvalue)

        if dilation[0] > 1 or dilation[1] > 1:  # only tensorflow support dilation
            n, c, h, w = inp2.shape
            kh, kw = kernel_size
            sh, sw = stride
            dh, dw = dilation

            dpb = math.ceil(h / dh) * dh - h
            dpr = math.ceil(w / dw) * dw - w
            inp_bk = torch.clone(inp2)
            inp2 = torch.nn.functional.pad(inp2, (0, dpr, 0, dpb), value=pvalue)

            nn, nc, nh, nw = inp2.shape
            inp2 = inp2.view(nn, nc, nh // dh, dh, nw // dw, dw)
            inp2 = inp2.permute(3, 5, 0, 1, 2, 4).contiguous()
            inp2 = inp2.view(nn * dw * dh, nc, nh // dh, nw // dw)

            y_sum = torch.nn.functional.avg_pool2d(inp2, kernel_size=(kh, kw), stride=(1, 1), padding=0,
                                                   count_include_pad=False, ceil_mode=ceil_mode, divisor_override=1)

            inp_n, inp_c, inp_h, inp_w = inp2.shape
            out_n, out_c, out_h, out_w = y_sum.shape
            y_area = torch.zeros_like(y_sum, device=inp.device)

            if count_include_pad:
                y_area = kernel_size[0] * kernel_size[1]
            else:
                h_start = (y_area + torch.arange(out_h, device=inp.device).reshape(out_h, 1)) * 1 - 0
                h_end = torch.min(h_start + kernel_size[0], y_area + inp_h)
                h_start = torch.max(h_start, y_area)
                w_start = (y_area + torch.arange(out_w, device=inp.device).reshape(1, out_w)) * 1 - 0
                w_end = torch.min(w_start + kernel_size[1], y_area + inp_w)
                w_start = torch.max(w_start, y_area)

                y_area = torch.multiply(h_end - h_start, w_end - w_start)
            if self.quantized:
                y_sum = torch.clamp(y_sum, -2 ** 31, 2 ** 31)
                do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
                    1. / y_area, mult_bits=8, force_shift_positive=False)
                y = linear_requantize(y_sum, do_scale, do_shift, self.outputs[0].zerop, out.qmin, out.qmax)
            else:
                y = y_sum / y_area

            cn, cc, ch, cw = y.shape
            y = y.view(dh, dw, cn // (dh * dw), cc, ch, cw)
            y = y.permute(2, 3, 4, 0, 5, 1).contiguous()
            y = y.view(cn // (dw * dh), cc, ch * dh, cw * dw)
            y = y[:, :, 0:ch * dh - dpb, 0:cw * dw - dpr]

        else:

            y_sum = torch.nn.functional.avg_pool2d(inp2,
                                                   kernel_size=kernel_size, stride=stride, padding=0,
                                                   count_include_pad=False, ceil_mode=ceil_mode,
                                                   divisor_override=1)

            inp_n, inp_c, inp_h, inp_w = inp.shape
            out_n, out_c, out_h, out_w = y_sum.shape
            y_area = torch.zeros_like(y_sum, device=inp.device)

            if count_include_pad:
                y_area = kernel_size[0] * kernel_size[1]
            else:
                h_start = (y_area + torch.arange(out_h, device=inp.device).reshape(out_h, 1)) * stride[0] - padding[2]
                h_end = torch.min(h_start + kernel_size[0], y_area + inp_h)
                h_start = torch.max(h_start, y_area)
                w_start = (y_area + torch.arange(out_w, device=inp.device).reshape(1, out_w)) * stride[1] - padding[0]
                w_end = torch.min(w_start + kernel_size[1], y_area + inp_w)
                w_start = torch.max(w_start, y_area)

                y_area = torch.multiply(h_end - h_start, w_end - w_start)
            if self.quantized:
                y_sum = torch.clamp(y_sum, -2 ** 31, 2 ** 31)
                do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
                    1. / y_area, mult_bits=8, force_shift_positive=False)
                y = linear_requantize(y_sum, do_scale, do_shift, self.outputs[0].zerop, out.qmin, out.qmax)
            else:
                y = y_sum / y_area

        self.outputs[0].betensor = nchw2nhwc(y)

    elif pmethod in ["L1", "L2"]:
        p = 1 if pmethod == "L1" else 2
        # TODO: support per-channel zerop and pad the per-channel zerop
        padding_value = -self.inputs[0].zerop[0] if self.quantized else 0
        zerop = self.inputs[0].zerop if self.quantized else 0
        inp = torch.nn.functional.pad(inp, padding, value=padding_value)
        inp += zerop
        batch = inp.shape[0]
        _, out_h, out_w, channel = self.outputs[0].ir_shape
        inp_h, inp_w = inp.shape[2], inp.shape[3]
        psum_placeholder = torch.zeros((batch, channel), device=inp.device)
        output = torch.zeros((batch, channel, out_h, out_w), device=inp.device)
        if pmethod == "L1":
            # Here abs is consistent with OnNX, but Torch doesn't need abs
            feature = torch.abs(inp)
        else:
            feature = inp * inp
        startX = torch.arange(out_w) * stride[1]
        startY = torch.arange(out_h) * stride[0]
        endY = startY + (kernel_size[0] - 1) * dilation[0] + 1
        endX = startX + (kernel_size[1] - 1) * dilation[1] + 1
        if ceil_mode:
            padding_right = max(0, endX.max() - inp_w)
            padding_bottom = max(0, endY.max() - inp_h)
            padding_diff = (0, padding_right, 0, padding_bottom)
            feature = torch.nn.functional.pad(feature, padding_diff, value=0)

        for r_idx in range(out_h):
            for c_idx in range(out_w):
                pool_value = feature[:, :, startY[r_idx]:endY[r_idx]:dilation[0], startX[c_idx]:endX[c_idx]:dilation[1]]
                poolsum = torch.sum(pool_value, dim=[2, 3])  # [1,32]
                if self.quantized:
                    shift = self.params["shift_value"]
                    scale = self.params["scale_value"]
                    act_qmin, act_qmax = bits2range(32, False)
                    poolsum = torch.clamp(poolsum, act_qmin, act_qmax)
                    if pmethod == "L1":
                        pool_output = linear_requantize(poolsum, scale, shift, out.zerop, out.qmin, out.qmax)
                    else:  # L2
                        pmin, pmax = dtype2range(Dtype.UINT16)
                        sqrt_lut = self.constants['sqrt_lut'].betensor
                        poolsum = linear_requantize(poolsum, scale, shift, 0, pmin, pmax).long()
                        poolsum2 = torch.reshape(poolsum, (-1,))
                        poolsqrt = lookup_lut_powerof2(poolsum2, sqrt_lut, 16, False, dtype2bits(
                            self.constants["sqrt_lut"].dtype), is_signed(self.constants["sqrt_lut"].dtype))
                        pool_output = torch.reshape(poolsqrt, poolsum.shape)
                else:
                    psum_placeholder = torch.cat((psum_placeholder, poolsum.reshape(batch, -1)), dim=-1)
                    pool_output = torch.pow(poolsum, 1/p)
                output[:, :, r_idx, c_idx] = pool_output
        out.betensor = nchw2nhwc(output)

        if not self.quantized:
            if len(self.placeholders) < 1:
                ph0 = PyTensor(self.name + "/power2_outputs",
                               psum_placeholder.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
                self.placeholders.append(ph0)
            self.placeholders[0].betensor = psum_placeholder
    else:
        OPT_WARN('Pooling Layer does not support method = %s' % pmethod)
    if int(out.betensor.shape[1]) != int(out.ir_shape[1]) or int(out.betensor.shape[2]) != int(out.ir_shape[2]):
        OPT_WARN("OPT inferred shape is mismatched with IR")

    # incase of pool1d
    if len(self.inputs[0].ir_shape) < 4:
        out.betensor = out.betensor.squeeze(1)

    out.betensor[out.betensor != out.betensor] = 0
    return out.betensor


@quant_register(OpType.Pooling)
def pooling_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_activation = self.attrs["q_bits_activation"]

    pmethod = self.get_param('method').upper()
    inp = self.inputs[0]
    out = self.outputs[0]

    if pmethod == 'L1':
        out_signed = False or self.force_dtype_int
        out.qbits = q_bits_activation
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, q_bits_activation, is_signed=out_signed)
        out.qinvariant = False

        do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
            out.scale / inp.scale, mult_bits=16, shift_bits_ceil=63)
        self.params["shift_value"] = int(do_shift)
        self.params["shift_type"] = do_shift_type
        self.params["scale_value"] = int(do_scale)
        self.params["scale_type"] = do_scale_type

    elif pmethod == 'L2':
        placeholders = self.placeholders[0]
        placeholders.scale, placeholders.zerop, placeholders.qmin, placeholders.qmax, placeholders.dtype = \
            get_linear_quant_params_from_tensor(
                placeholders, QuantMode.to_symmetric(q_mode_activation), 16, is_signed=False)

        out.qbits = q_bits_activation
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, q_bits_activation, is_signed=False or self.force_dtype_int)
        out.qinvariant = False

        do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
            placeholders.scale / (inp.scale * inp.scale), mult_bits=16, shift_bits_ceil=63)

        lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
        lut = linear_dequantize(torch.linspace(
            placeholders.qmin, placeholders.qmax, steps=lsteps), placeholders.scale, 0)
        lut = torch.sqrt(lut)
        lut = linear_quantize_clip(
            lut, out.scale, out.zerop, out.qmin, out.qmax)

        self.constants["sqrt_lut"] = PyTensor(
            self.name + "/sqrt_lut", lut.cpu().numpy().astype(dtype2nptype(out.dtype)))

        self.params["shift_value"] = int(do_shift)
        self.params["shift_type"] = do_shift_type
        self.params["scale_value"] = int(do_scale)
        self.params["scale_type"] = do_scale_type
    else:
        out.scale = inp.scale
        out.zerop = inp.zerop
        out.dtype = inp.dtype
        out.qmin = inp.qmin
        out.qmax = inp.qmax
        out.qbits = inp.qbits
        out.qinvariant = inp.qinvariant
