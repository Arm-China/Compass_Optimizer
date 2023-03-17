# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch.nn.functional as Func

'''
align_corners and asymmetric are mutually exclusive.

| half_pixel | algin_corners | asymmetric|
|----------------------------------------|
| True       | False         | True      |
| False      | False         | True      |
| False      | True          | False     |

| align_corners | asymmetric | half_pixel|
|----------------------------------------|
| True       | False         | False     |
| False      | True          | True      |
| False      | True          | False     |

'''


def _scaler(i, ratio, mode, input_size, out_size):
    if mode == 'half_pixel':
        return (i + 0.5) * (1. / ratio) - 0.5 if ratio is not None else (i + 0.5) * (input_size / out_size) - 0.5
    elif mode == 'align_corners':
        return i * (input_size - 1) / (out_size - 1)
    elif mode == 'pytorch_half_pixel':
        if out_size > 1:
            return (i + 0.5) * (1. / ratio) - 0.5 if ratio is not None else (i + 0.5) * (input_size / out_size) - 0.5
        else:
            return 0
    elif mode == 'tf_half_pixel_for_nn':
        return (i + 0.5) * (1. / ratio) if ratio is not None else (i + 0.5) * (input_size / out_size)
    else:  # asymmetric
        return i * (1. / ratio) if ratio is not None else i * (input_size / out_size)


def accelerate_resize_bilinear(in_data, output_shape, mode, ratio_x, ratio_y, coordination_shift):
    batch, out_h, out_w, out_c = output_shape
    in_h, in_w = in_data.shape[1:3]
    scale_num = 2 ** coordination_shift
    quantized = False if coordination_shift == 0 else True
    y = _scaler(torch.arange(out_h, device=in_data.device), ratio_y, mode, in_h, out_h)
    x = _scaler(torch.arange(out_w, device=in_data.device), ratio_x, mode, in_w, out_w)

    y_floor = torch.floor(y).int()
    y_ceil = torch.ceil(y).int()
    x_floor = torch.floor(x).int()
    x_ceil = torch.ceil(x).int()

    y0 = torch.maximum(y_floor, torch.tensor(0, device=in_data.device))
    y1 = torch.minimum(y_ceil, torch.tensor(in_h - 1, device=in_data.device))
    x0 = torch.maximum(x_floor, torch.tensor(0, device=in_data.device))
    x1 = torch.minimum(x_ceil, torch.tensor(in_w - 1, device=in_data.device))

    y0_x0 = torch.cartesian_prod(y0, x0).long()
    y0_x1 = torch.cartesian_prod(y0, x1).long()
    y1_x0 = torch.cartesian_prod(y1, x0).long()
    y1_x1 = torch.cartesian_prod(y1, x1).long()

    f00 = in_data[:, y0_x0[:, 0], y0_x0[:, 1], :]
    f01 = in_data[:, y0_x1[:, 0], y0_x1[:, 1], :]
    f10 = in_data[:, y1_x0[:, 0], y1_x0[:, 1], :]
    f11 = in_data[:, y1_x1[:, 0], y1_x1[:, 1], :]

    y0_lerp = y - y0
    x0_lerp = x - x0
    y1_lerp = 1 - y0_lerp
    x1_lerp = 1 - x0_lerp

    y0l_x0l = torch.cartesian_prod(y0_lerp, x0_lerp)
    y0l_x1l = torch.cartesian_prod(y0_lerp, x1_lerp)
    y1l_x0l = torch.cartesian_prod(y1_lerp, x0_lerp)
    y1l_x1l = torch.cartesian_prod(y1_lerp, x1_lerp)

    Q00 = y1l_x1l[:, 0] * y1l_x1l[:, 1] * scale_num
    Q10 = y0l_x1l[:, 0] * y0l_x1l[:, 1] * scale_num
    Q01 = y1l_x0l[:, 0] * y1l_x0l[:, 1] * scale_num
    Q11 = y0l_x0l[:, 0] * y0l_x0l[:, 1] * scale_num
    if quantized:
        Q00 = torch.floor(Q00 + 0.5)
        Q10 = torch.floor(Q10 + 0.5)
        Q01 = torch.floor(Q01 + 0.5)
        Q11 = torch.floor(Q11 + 0.5)

    Q00 = Q00.reshape([-1, 1]).repeat(batch, 1, out_c)
    Q01 = Q01.reshape([-1, 1]).repeat(batch, 1, out_c)
    Q10 = Q10.reshape([-1, 1]).repeat(batch, 1, out_c)
    Q11 = Q11.reshape([-1, 1]).repeat(batch, 1, out_c)
    bilinear_out = (f00 * Q00 + f01 * Q01 + f10 * Q10 + f11 * Q11) / scale_num
    bilinear_out = torch.round(bilinear_out) if quantized else bilinear_out
    outt = bilinear_out.reshape(*output_shape)
    return outt


def resize_bilinear(input_data, params):
    mode = params['mode']
    output_shape = params['output_shape']
    coordination_shift = params['coordination_shift'] if 'coordination_shift' in params else 0
    align_corners = False
    half_pixel = False
    ratio_x = params['ratio_x']
    ratio_y = params['ratio_y']

    out = accelerate_resize_bilinear(input_data,
                                     output_shape,
                                     mode,
                                     ratio_x,
                                     ratio_y,
                                     coordination_shift)
    return out


def TF1_compatible_resize(inp, output_shape, mode):
    #############for bilinear interploration#############
    # align_corners = False
    # x_ori is the coordinate in original image
    # x_up is the coordinate in the upsampled image
    # x_ori = (x_up + 0.5) / factor - 0.5

    # # align_corners = True
    # # h_ori is the height in original image
    # # h_up is the height in the upsampled image
    # stride = (h_ori - 1) / (h_up - 1)
    # x_ori_list = []
    # # append the first coordinate
    # x_ori_list.append(0)
    # for i in range(1, h_up - 1):
    #     x_ori_list.append(0 + i * stride)
    # # append the last coordinate
    # x_ori_list.append(h_ori - 1)

    ###############for nearest mode with align_corner = True############
    batch = inp.shape[0]
    inp_h = inp.shape[2]
    inp_w = inp.shape[3]

    out_h = output_shape[1]
    out_w = output_shape[2]
    out_c = output_shape[3]

    if mode == 'align_corners':
        factor_h = max((out_h - 1), 1) / max((inp_h - 1), 1)
        factor_w = max((out_w - 1), 1) / max((inp_w - 1), 1)
    else:  # mode == 'asymmetric'
        factor_h = max(out_h, 1) / max(inp_h, 1)
        factor_w = max(out_w, 1) / max(inp_w, 1)

    out_t = torch.zeros([batch, out_c, out_h, out_w])

    import math
    for h in range(out_h):
        for w in range(out_w):
            # round and math.floor is align with the tf implementation
            map_inp_h = min(int(round(h / factor_h) if mode == 'align_corners' else math.floor(h / factor_h)),
                            inp_h - 1)
            map_inp_w = min(int(round(w / factor_w) if mode == 'align_corners' else math.floor(w / factor_w)),
                            inp_w - 1)
            out_t[:, :, h, w] = inp[:, :, map_inp_h, map_inp_w]
    return out_t


def nearest_resize(input_data, output_shape, mode, ratio_x, ratio_y, nearest_mode):
    batch, out_h, out_w, out_c = output_shape
    in_h, in_w = input_data.shape[1:3]

    def get_pixel(nearest_mode, original_pixel, down_sample):
        if nearest_mode == "round_prefer_floor":
            mask = torch.eq(original_pixel, original_pixel.int() + 0.5)
            return torch.where(mask, torch.floor(original_pixel).int(), torch.round(original_pixel).int())
        elif nearest_mode == "floor":
            return torch.floor(original_pixel).int()
        elif nearest_mode == "ceil":
            return torch.ceil(original_pixel).int()
        elif nearest_mode == "simple":
            return torch.ceil(original_pixel).int() if down_sample else original_pixel.int()
        else:  # "round_prefer_ceil"
            return torch.floor(original_pixel + 0.5).int()

    y = _scaler(torch.arange(out_h, device=input_data.device),
                ratio_y, mode, in_h, out_h)
    x = _scaler(torch.arange(out_w, device=input_data.device),
                ratio_x, mode, in_w, out_w)

    map_inp_h = get_pixel(nearest_mode, y, out_h / in_h < 1)
    map_inp_w = get_pixel(nearest_mode, x, out_w / in_w < 1)

    map_inp_h = torch.maximum(
        map_inp_h, torch.tensor(0, device=input_data.device))
    map_inp_h = torch.minimum(map_inp_h, torch.tensor(
        in_h - 1, device=input_data.device))
    map_inp_w = torch.maximum(
        map_inp_w, torch.tensor(0, device=input_data.device))
    map_inp_w = torch.minimum(map_inp_w, torch.tensor(
        in_w - 1, device=input_data.device))

    input_index = torch.cartesian_prod(map_inp_h, map_inp_w).long()

    out_t = input_data[:, input_index[:, 0],
                       input_index[:, 1], :].reshape(output_shape)

    return out_t


@op_register(OpType.Interp)
def interp(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    method = self.get_param('method').lower()
    rmode = self.get_param('mode').lower()
    ratio_x = self.get_param('ratio_x', optional=True, default_value=None)
    ratio_y = self.get_param('ratio_y', optional=True, default_value=None)
    output_shape = out.ir_shape

    _supported_method = ['nearest', 'linear',
                         'bilinear', 'bicubic', 'trilinear', 'area']
    if method not in _supported_method:
        OPT_WARN(
            f"please check the method of Resize op, which now is {method}, but Optimizer only supports {_supported_method}, now use 'bilinear' instead of {method}")
        method = 'bilinear'

    _supported_mode = ['half_pixel', 'align_corners',
                       'asymmetric', 'pytorch_half_pixel', 'tf_half_pixel_for_nn']
    if rmode not in _supported_mode:
        OPT_WARN(
            f"please check the mode of Resize op, which now is {rmode}, but Optimizer only supports {_supported_mode}, now use 'half_pixel' instead of {rmode}")
        rmode = 'half_pixel'

    if method == 'bilinear':
        params = {}
        params['mode'] = rmode
        params['ratio_x'] = ratio_x
        params['ratio_y'] = ratio_y
        params['output_shape'] = [inp.shape[0]] + list(output_shape)[1:]
        if self.quantized:
            coordination_shift = self.get_param('interp_shift_value')
            params['coordination_shift'] = coordination_shift
        input_data = inp.betensor.float()
        outp = resize_bilinear(input_data, params)
    elif method in ['linear', 'bicubic', 'trilinear']:
        # TODO: support mode == 'asymmetric'
        inpt = nhwc2nchw(inp.betensor).float()
        outp = Func.interpolate(
            inpt, output_shape[1:3], mode=method, align_corners=(rmode == 'align_corners'))
        outp = nchw2nhwc(outp)
    else:  # method==nearest
        # TODO Need align with IR define
        n_mode = self.get_param(
            'nearest_mode', optional=True, default_value='round_prefer_floor').lower()
        inpt = inp.betensor.float()
        output_shape = [inp.shape[0]] + list(output_shape)[1:]
        outp = nearest_resize(inpt, output_shape, rmode,
                              ratio_x, ratio_y, n_mode)

    if self.quantized:
        outp = torch.clamp(torch.round(outp), out.qmin, out.qmax)
    out.betensor = outp
    return out.betensor


@quant_register(OpType.Interp)
def interp_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    if self.attrs['resize_degrade_to_nearest']:
        self.params['method'] = 'nearest'
        self.attrs['optimization_info']['resize_degrade_to_nearest'] = True

    interp_shift = 0
    if self.get_param('method').lower() == 'bilinear':
        # interp_shift = 13
        interp_shift = self.attrs['scaling_bits'][0]  # default value=13
    self.params['interp_shift_value'] = interp_shift
    self.params['interp_shift_type'] = SHIFT_DTYPE
