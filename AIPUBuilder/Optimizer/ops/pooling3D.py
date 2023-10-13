# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch
import math


@op_register(OpType.Pooling3D)
def pooling3D(self, *args):
    """
    D: Depth, H: Height, W: Width, C: Channel
    :param self:  input size(N,D,H,W,C), kernel_size(kD,kH,kW), stride(sD,sH,sW), padding([p_head, p_tail],[p_left, p_right],[p_top,p_bottom]
    :param args:
    :return: output:(N,Dout,Hout,Wout,C)
    """
    _METHOD_TO_FUNC = {'MAX': torch.nn.functional.max_pool3d, 'AVG': torch.nn.functional.avg_pool3d}
    method = self.get_param('method').upper()
    if method not in _METHOD_TO_FUNC.keys():
        OPT_FATAL(f"id={self.attrs['layer_id']},op_type={str(self.type)} DoesNot support:{method},"
                  f" only support:{str(_METHOD_TO_FUNC.keys())}")
    pool3d_func = _METHOD_TO_FUNC[method]

    out = self.outputs[0]
    inp = self.inputs[0].betensor.float()

    kernel_size = self.get_param('kernel_z'), self.get_param('kernel_y'), self.get_param('kernel_x')
    stride = self.get_param('stride_z'), self.get_param('stride_y'), self.get_param('stride_x')
    padding = (self.get_param('pad_x_begin'), self.get_param('pad_x_end'),
               self.get_param('pad_y_begin'), self.get_param('pad_y_end'),
               self.get_param('pad_z_begin'), self.get_param('pad_z_end'))

    # compare with output height to determine ceil_mode
    out_sh = math.floor((inp.shape[1] - kernel_size[0] + padding[2] + padding[3]) / stride[0]) + 1
    local_ceil_mode = False if out_sh == out.ir_shape[1] else True
    original_ceil_mode = self.get_param('ceil_mode', optional=True, default_value=local_ceil_mode)
    ceil_mode = original_ceil_mode
    count_include_pad = self.get_param('count_include_pad', optional=True, default_value=True)
    dilation = self.get_param('dilation_z'), self.get_param('dilation_y'), self.get_param('dilation_x')
    inpp = inp.permute(0, 4, 1, 2, 3)
    if method == 'MAX':
        pvalue = torch.finfo(torch.float32).min if not self.quantized else OPT_INT_MIN
    else:
        pvalue = 0  # (0 - out.zerop) if self.quantized and count_include_pad else 0 # lib's logic

    # oz=math.ceil((inp.shape[1]+padding[4]+padding[5]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1 if ceil_mode==True \
    #     else math.floor((inp.shape[1]+padding[4]+padding[5]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
    oz = self.outputs[0].ir_shape[1]
    extra_pz = (oz - 1) * stride[0] + dilation[0] * (kernel_size[0] - 1) + 1 - inp.shape[1] - padding[4] - padding[5]

    # oh=math.ceil((inp.shape[2]+padding[2]+padding[3]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1 if ceil_mode==True \
    #     else math.floor((inp.shape[2]+padding[2]+padding[3]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
    oh = self.outputs[0].ir_shape[2]
    extra_ph = (oh - 1) * stride[1] + dilation[1] * (kernel_size[1] - 1) + 1 - inp.shape[2] - padding[2] - padding[3]

    # ow=math.ceil((inp.shape[3]+padding[0]+padding[1]-dilation[2]*(kernel_size[2]-1)-1)/stride[2])+1 if ceil_mode==True \
    #     else math.floor((inp.shape[3]+padding[0]+padding[1]-dilation[2]*(kernel_size[2]-1)-1)/stride[2])+1
    ow = self.outputs[0].ir_shape[3]
    extra_pw = (ow - 1) * stride[2] + dilation[2] * (kernel_size[2] - 1) + 1 - inp.shape[3] - padding[0] - padding[1]

    inp = torch.nn.functional.pad((inpp + self.inputs[0].zerop) if self.quantized else inpp, padding, value=pvalue)
    if (extra_pz > 0 or extra_ph > 0 or extra_pw > 0) and original_ceil_mode:
        ceil_mode = False
        inp = torch.nn.functional.pad(inp, (0, extra_pw, 0, extra_ph, 0, extra_pz), value=pvalue)

    if extra_pz + padding[5] > kernel_size[0] and ceil_mode:
        OPT_WARN('ceil_mode=True cause total padding_z exceed kernel_z. Result maybe untrustworthy.')
    if extra_ph + padding[3] > kernel_size[1] and ceil_mode:
        OPT_WARN('ceil_mode=True cause total padding_h exceed kernel_h. Result maybe untrustworthy.')
    if extra_pw + padding[1] > kernel_size[2] and ceil_mode:
        OPT_WARN('ceil_mode=True cause total padding_w exceed kernel_w. Result maybe untrustworthy.')

    if not self.quantized or method == 'MAX':
        if method == 'MAX':
            outt = pool3d_func(inp,
                               kernel_size,
                               stride,
                               padding=0,
                               dilation=dilation,
                               ceil_mode=ceil_mode)
            if self.quantized:
                outt = torch.clamp(outt, out.qmin, out.qmax)
        else:  # avg
            outt = pool3d_func(inp,
                               kernel_size,
                               stride,
                               padding=0,
                               ceil_mode=ceil_mode,
                               count_include_pad=count_include_pad)
            if not count_include_pad:  # compensate padding value mismatch
                pad_2 = torch.ones_like(inpp)
                pad_2 = torch.nn.functional.pad(pad_2, padding, value=pvalue)
                if (extra_pz > 0 or extra_ph > 0 or extra_pw > 0) and original_ceil_mode:
                    ceil_mode = False
                    pad_2 = torch.nn.functional.pad(pad_2, (0, extra_pw, 0, extra_ph, 0, extra_pz),
                                                    value=pvalue)
                one_div = torch.nn.functional.avg_pool3d(pad_2, kernel_size=kernel_size, stride=stride, padding=0,
                                                         ceil_mode=ceil_mode, count_include_pad=False)
                kernel_prod = kernel_size[0] * kernel_size[1] * kernel_size[2]
                div_mask = one_div * kernel_prod  # mask sum up
                outt = outt * kernel_prod / div_mask

    else:  # method='AVG' and self.quantized
        # lib implementation by using torch
        y_sum = pool3d_func(inp,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=0,
                            ceil_mode=ceil_mode,
                            count_include_pad=False,
                            divisor_override=1)
        y_sum = torch.clamp(y_sum, -2 ** 31, 2 ** 31 - 1)
        outt = y_sum
        out_n, out_c, out_d, out_h, out_w = y_sum.shape
        inp_n, inp_c, inp_d, inp_h, inp_w = inp.shape
        '''
        # best precision
        y_area = torch.zeros_like(y_sum, device=inp.device)
        d_start = torch.max(
            (y_area + torch.arange(out_d, device=inp.device).reshape(out_d, 1, 1)) * stride[0] - padding[2], y_area)
        d_end = torch.min(d_start + kernel_size[0], y_area + inp_d)
        h_start = torch.max(
            (y_area + torch.arange(out_h, device=inp.device).reshape(out_h, 1)) * stride[1] - padding[0], y_area)
        h_end = torch.min(h_start + kernel_size[1], y_area + inp_h)
        w_start = torch.max(
            (y_area + torch.arange(out_w, device=inp.device).reshape(1, out_w)) * stride[2] - padding[1], y_area)
        w_end = torch.min(w_start + kernel_size[2], y_area + inp_w)
        y_area = torch.multiply(torch.multiply(h_end - h_start, w_end - w_start), d_end - d_start)
        do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(1. / y_area, mult_bits=8, force_shift_positive=False)
        outt = torch.multiply(torch.multiply(y_sum, do_scale), torch.pow(0.5, do_shift))
        outt = torch.clamp(torch.round(outt), -2**31,2**31)

        '''

        if not count_include_pad:  # compensate padding value mismatch
            pad_2 = torch.ones_like(inpp)
            pad_2 = torch.nn.functional.pad(pad_2, padding, value=pvalue)
            if (extra_pz > 0 or extra_ph > 0 or extra_pw > 0) and original_ceil_mode:
                ceil_mode = False
                pad_2 = torch.nn.functional.pad(pad_2, (0, extra_pw, 0, extra_ph, 0, extra_pz),
                                                value=pvalue)
            one_div = torch.nn.functional.avg_pool3d(pad_2, kernel_size=kernel_size, stride=stride, padding=0,
                                                     ceil_mode=ceil_mode, count_include_pad=False)
            kernel_prod = kernel_size[0] * kernel_size[1] * kernel_size[2]
            div_mask = one_div * kernel_prod  # mask sum up
            outt = outt * kernel_prod / div_mask

        # stage1: for kh*kw
        y_area = torch.zeros_like(y_sum, device=inp.device)
        h_start = torch.max(
            (y_area + torch.arange(out_h, device=inp.device).reshape(out_h, 1)) * stride[1] - padding[0], y_area)
        h_end = torch.min(h_start + kernel_size[1], y_area + inp_h)
        w_start = torch.max(
            (y_area + torch.arange(out_w, device=inp.device).reshape(1, out_w)) * stride[2] - padding[1], y_area)
        w_end = torch.min(w_start + kernel_size[2], y_area + inp_w)
        y_area = torch.multiply(h_end - h_start, w_end - w_start)
        do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(1. / y_area, mult_bits=8,
                                                                                          force_shift_positive=False)
        outt = torch.multiply(torch.multiply(outt, do_scale), torch.pow(0.5, do_shift))
        outt = torch.clamp(torch.round(outt), -2 ** 31, 2 ** 31)

        # stage2: for kd
        y_area = torch.zeros_like(y_sum, device=inp.device)
        d_start = torch.max(
            (y_area + torch.arange(out_d, device=inp.device).reshape(out_d, 1, 1)) * stride[0] - padding[2], y_area)
        d_end = torch.min(d_start + kernel_size[0], y_area + inp_d)
        y_area = d_end - d_start
        do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(1. / y_area, mult_bits=8,
                                                                                          force_shift_positive=False)
        outt = torch.multiply(torch.multiply(outt, do_scale), torch.pow(0.5, do_shift))
        outt = torch.clamp(torch.round(outt), -2 ** 31, 2 ** 31)

        outt = torch.clamp(torch.round(outt), out.qmin, out.qmax)

    outt = outt.permute(0, 2, 3, 4, 1)
    outt[outt != outt] = 0
    out.betensor = (outt - self.outputs[0].zerop) if self.quantized else outt
    return out.betensor


@quant_register(OpType.Pooling3D)
def pooling3D_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.dtype = inp.dtype
    out.qbits = inp.qbits
    out.qmin = inp.qmin
    out.qmax = inp.qmax
    out.qinvariant = inp.qinvariant
