# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_FATAL
import torch

register_optype('AdaptivePool')


@op_register(OpType.AdaptivePool)
def adaptive(self, *args):
    method = self.get_param('method').upper()
    inp = self.inputs[0].betensor
    out = self.outputs[0]

    if inp.ndim > 4 or inp.ndim < 3:
        OPT_FATAL(f"{self}, currently input ndim only support 3 or 4, such [N,H,W,C] or [H,W,C], please check! ")
    SUPPORT_FUNC = {
        "AVG": torch.nn.AdaptiveAvgPool2d,
        "MAX": torch.nn.AdaptiveMaxPool2d,
    }
    if method not in SUPPORT_FUNC:
        OPT_FATAL(f"{self}, currently method only support {SUPPORT_FUNC}, please check! ")

    is4dim = True if inp.ndim == 4 else False
    in_h, in_w = inp.shape[1:3] if is4dim else inp.shape[0:2]
    out_h, out_w = self.get_param('output_size')  # out.ir_shape[1:3] if is4dim else out.ir_shape[0:2]

    if self.quantized:
        channel = inp.shape[-1]
        out_shape = (inp.shape[0], out_h, out_w, channel) if is4dim else (1, out_h, out_w, channel)
        if not is4dim:
            inp = torch.unsqueeze(inp, dim=0)
        output = torch.zeros(out_shape, device=inp.device)

        h_start = torch.arange(out_h, device=inp.device).int() // out_h * in_h + (
            torch.arange(out_h, device=inp.device).int() % out_h) * in_h // out_h
        w_start = torch.arange(out_w, device=inp.device).int() // out_w * in_w + (
            torch.arange(out_w, device=inp.device).int() % out_w) * in_w // out_w
        h_end = 1 + ((torch.arange(out_h, device=inp.device).int() + 1) * in_h - 1) // out_h
        w_end = 1 + ((torch.arange(out_w, device=inp.device).int() + 1) * in_w - 1) // out_w
        if method == 'AVG':
            kenel_dict = {}
            for oh in range(out_h):
                for ow in range(out_w):
                    tmp_data = inp[:, h_start[oh]:h_end[oh], w_start[ow]:w_end[ow], :]
                    tmp_sum = torch.sum(tmp_data, dim=[1, 2])
                    kenel_area = (h_end[oh] - h_start[oh]) * (w_end[ow] - w_start[ow])
                    if kenel_area not in kenel_dict:
                        do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
                            1. / kenel_area,
                            mult_bits=8,
                            force_shift_positive=False)
                        kenel_dict[kenel_area] = (do_scale, do_shift)
                    else:
                        do_scale, do_shift = kenel_dict[kenel_area]
                    output[:, oh, ow, :] = linear_requantize(tmp_sum, do_scale, do_shift, 0, out.qmin, out.qmax)
        if method == 'MAX':
            for oh in range(out_h):
                for ow in range(out_w):
                    tmp_data = inp[:, h_start[oh]:h_end[oh], w_start[ow]:w_end[ow], :]
                    max_value = torch.amax(tmp_data, dim=[1, 2])
                    output[:, oh, ow, :] = max_value

        out.betensor = output if is4dim else output[0]
    else:
        func = SUPPORT_FUNC[method]((out_h, out_w))
        inp = nhwc2nchw(inp) if is4dim else torch.permute(inp, [2, 0, 1])
        output = func(inp)
        out.betensor = nchw2nhwc(output) if is4dim else torch.permute(output, [1, 2, 0])

    return out.betensor


@quant_register(OpType.AdaptivePool)
def adaptive_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.qinvariant = inp.qinvariant
