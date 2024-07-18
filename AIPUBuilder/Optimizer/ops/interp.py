# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

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
        return i * 0. if out_size == 1 else (i * (input_size - 1) / (out_size - 1))
    elif mode == 'pytorch_half_pixel':
        if out_size > 1:
            return (i + 0.5) * (1. / ratio) - 0.5 if ratio is not None else (i + 0.5) * (input_size / out_size) - 0.5
        else:
            return i * 0.
    elif mode == 'tf_half_pixel_for_nn':
        return (i + 0.5) * (1. / ratio) if ratio is not None else (i + 0.5) * (input_size / out_size)
    elif mode == 'half_pixel_symmetric':
        if ratio is not None:
            adj = out_size / (input_size * ratio)
            center = input_size / 2.
            offset = center * (1. - adj)
            return offset + (i + 0.5) * (1. / ratio) - 0.5
        else:
            return (i + 0.5) * (input_size / out_size) - 0.5
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
    ratio_x = params['ratio_x']
    ratio_y = params['ratio_y']

    out = accelerate_resize_bilinear(input_data,
                                     output_shape,
                                     mode,
                                     ratio_x,
                                     ratio_y,
                                     coordination_shift)
    return out


def compute_weight_coefficients(input_size, output_size, rscale, mode, exclude_outside, dev):
    def filter(x):
        if x < 0.0:
            x = -x
        if x < 1.0:
            return 1.0 - x
        return 0.0
    import math
    scale = 1/rscale
    support = scale if (scale >= 1.0) else 1.0
    windowsize = int(math.ceil(support)) * 2 + 1
    inv_scale = 1/(scale) if scale >= 1.0 else 1.0
    bound_w_min = []
    bound_w_max = []
    scale_buffer = torch.zeros([windowsize * output_size], device=dev)
    for coord in range(output_size):
        center = 0.5 + _scaler(coord, rscale, mode, input_size, output_size)
        total_weight = 0.0
        fmin = math.floor(center - support + 0.5)
        fmax = math.floor(center + support + 0.5)
        xmin_real = int(fmin)
        xmax_real = int(fmax)
        xmin_cut = max(0, xmin_real)
        xmax_cut = min(input_size, xmax_real)
        bound_w_min.append(xmin_cut)
        bound_w_max.append(xmax_cut)
        xmin = xmin_cut if exclude_outside else xmin_real
        xmax = xmax_cut if exclude_outside else xmax_real
        terp = xmax - xmin
        offset = coord*windowsize
        for x in range(terp):
            weight = filter((x + xmin - center + 0.5) * inv_scale)
            scale_buffer[offset + x] = weight
            total_weight += weight
        xmax -= xmin
        if not exclude_outside:
            neg_xsize = -xmin if xmin < 0 else 0
            for i in range(neg_xsize):
                scale_buffer[offset + neg_xsize] += scale_buffer[offset + i]
            bound_xsize = (xmax + xmin - input_size) if (xmax + xmin > input_size) else 0
            for x in range(xmax - bound_xsize, xmax):
                scale_buffer[offset+xmax - bound_xsize - 1] += scale_buffer[offset + x]
            x = 0
            while (neg_xsize | bound_xsize) > 0 and (x < xmax_cut - xmin_cut):
                scale_buffer[offset + x] = scale_buffer[offset + x + neg_xsize]
                x += 1
        total_weight_inv = 1.0 if total_weight == 0.0 else 1.0 / total_weight
        for x in range(xmax_cut - xmin_cut):
            scale_buffer[offset + x] *= total_weight_inv
    return scale_buffer, torch.tensor(bound_w_min, device=dev), torch.tensor(bound_w_max, device=dev)


def resize_bilinear_antialias(self, input_data, params):
    mode = params['mode']
    output_shape = params['output_shape']
    exclude_outside = params['exclude_outside']
    coordination_shift = params['coordination_shift'] if 'coordination_shift' in params else 0
    quantized = False if coordination_shift == 0 else True
    ratio_x = params['ratio_x']
    ratio_y = params['ratio_y']
    dev = input_data.device
    input_height, input_width = input_data.shape[1:3]
    batch_size, output_height, output_width, channel = output_shape
    output_data = torch.zeros([batch_size, output_height, output_width, channel], device=input_data.device)
    if 'weight_coefficients_x' not in self.constants:
        weight_coefficients_x, bound_x_min, bound_x_max = compute_weight_coefficients(
            input_width, output_width, ratio_x, mode, exclude_outside, dev)
        weight_coefficients_y, bound_y_min, bound_y_max = compute_weight_coefficients(
            input_height, output_height, ratio_y, mode, exclude_outside, dev)
        self.constants["weight_coefficients_x"] = PyTensor(
            self.name+"/weight_coefficients_x", weight_coefficients_x.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
        self.constants["weight_coefficients_y"] = PyTensor(
            self.name+"/weight_coefficients_y", weight_coefficients_y.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
        # currently set to uint16
        bound_x_dtype = bound_y_dtype = bits2dtype(16, False)
        # _, bound_x_dtype = range2dtype(0, input_width)
        # _, bound_y_dtype = range2dtype(0, input_height)
        self.constants["bound_x_min"] = PyTensor(
            self.name+"/bound_x_min", bound_x_min.cpu().numpy().astype(dtype2nptype(bound_x_dtype)))
        self.constants["bound_x_max"] = PyTensor(
            self.name+"/bound_x_max", bound_x_max.cpu().numpy().astype(dtype2nptype(bound_x_dtype)))
        self.constants["bound_y_min"] = PyTensor(
            self.name+"/bound_y_min", bound_y_min.cpu().numpy().astype(dtype2nptype(bound_y_dtype)))
        self.constants["bound_y_max"] = PyTensor(
            self.name+"/bound_y_max", bound_y_max.cpu().numpy().astype(dtype2nptype(bound_y_dtype)))
    else:
        weight_coefficients_x = self.constants['weight_coefficients_x'].betensor
        weight_coefficients_y = self.constants['weight_coefficients_y'].betensor
        bound_x_min = self.constants['bound_x_min'].betensor
        bound_x_max = self.constants['bound_x_max'].betensor
        bound_y_min = self.constants['bound_y_min'].betensor
        bound_y_max = self.constants['bound_y_max'].betensor
    windowsize_w = weight_coefficients_x.shape[0] // output_width
    windowsize_h = weight_coefficients_y.shape[0] // output_height
    image_temp_buffer = torch.zeros([batch_size, input_height, output_width, channel], device=input_data.device)

    ############################################forward##################################################
    # horizon interpolate
    for w in range(output_width):
        xmin = bound_x_min[w]
        xmax = bound_x_max[w]
        tmp_data = (input_data[:, :, xmin:xmax, :] * weight_coefficients_x[w *
                    windowsize_w: w*windowsize_w + xmax - xmin].reshape([1, 1, -1, 1]))
        output = torch.sum(tmp_data, dim=2, keepdim=False)
        image_temp_buffer[:, :, w, :] = output
    if quantized:
        image_temp_buffer = linear_requantize(image_temp_buffer, 1, coordination_shift,
                                              0, self.inputs[0].qmin, self.inputs[0].qmax)

    # vertical interpolate
    for h in range(output_height):
        xmin = bound_y_min[h]
        xmax = bound_y_max[h]
        tmp_data = (image_temp_buffer[:, xmin:xmax, :, :] * weight_coefficients_y[h *
                    windowsize_h: h*windowsize_h + xmax - xmin].reshape([1, -1, 1, 1]))
        output = torch.sum(tmp_data, dim=1, keepdim=False)
        output_data[:, h, :, :] = output
    if quantized:
        output_data = linear_requantize(output_data, 1, coordination_shift, 0, self.inputs[0].qmin, self.inputs[0].qmax)
    ############################################forward###################################################
    return output_data


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
            mask = torch.isclose(original_pixel, original_pixel.int() + 0.5, rtol=1e-07, atol=1e-09)
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
    ratio_x = self.get_param('ratio_x', optional=True,
                             default_value=self.outputs[0].ir_shape[2]/self.inputs[0].ir_shape[2])
    ratio_y = self.get_param('ratio_y', optional=True,
                             default_value=self.outputs[0].ir_shape[1]/self.inputs[0].ir_shape[1])
    antialias = self.get_param('antialias', optional=True, default_value=False)
    exclude_outside = self.get_param('exclude_outside', optional=True, default_value=False)
    output_shape = out.ir_shape
    _supported_method = ['nearest', 'linear',
                         'bilinear', 'bicubic', 'trilinear', 'area']
    if method not in _supported_method:
        OPT_WARN(
            f"please check the method of Resize op, which now is {method}, but Optimizer only supports {_supported_method}, now use 'bilinear' instead of {method}")
        method = 'bilinear'

    if antialias:
        if method not in ['bilinear']:
            OPT_WARN(
                f"antialias is only valid for bilinear now (will support bicubic antialias in feature), but method is {method}, so we reset antialias false")
            antialias = False
            self.params['antialias'] = antialias
        if ratio_y > 1.0 or ratio_x > 1.0:
            OPT_WARN(
                f"antialias is only valid for downscaling, so we reset antialias false")
            antialias = False
            self.params['antialias'] = antialias

    _supported_mode = ['half_pixel', 'align_corners',
                       'asymmetric', 'pytorch_half_pixel', 'tf_half_pixel_for_nn', 'half_pixel_symmetric']
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
        input_data = inp.betensor.float()
        if self.quantized:
            coordination_shift = self.get_param('interp_shift_value')
            params['coordination_shift'] = coordination_shift
        if antialias:
            params['exclude_outside'] = exclude_outside
            outp = resize_bilinear_antialias(self, input_data, params)
        else:
            outp = resize_bilinear(input_data, params)

    elif method in ['linear', 'bicubic', 'trilinear']:
        # TODO: support mode == 'asymmetric'
        # TODO: support bicubic antialias
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
    extra_params = self.get_attrs('extra_params', optional=True, default_value=[0, 13])  # default value=13
    interp_shift = 0
    if self.get_param('method').lower() == 'bilinear':
        antialias = self.get_param('antialias', optional=True, default_value=False)
        if antialias:
            if 'extra_params' not in self.attrs:
                # default value = 8
                interp_shift = 8
            else:
                interp_shift = extra_params[1]
            multiplier = (2 ** interp_shift) - 1
            _, input_height, input_width, _ = self.inputs[0].ir_shape
            _, output_height, output_width, _ = self.outputs[0].ir_shape
            ratio_y = output_height / input_height
            ratio_x = output_width / input_width
            mode = self.get_param('mode').lower()
            exclude_outside = self.get_param('exclude_outside', optional=True, default_value=False)
            dev = self.inputs[0].betensor.device
            weight_coefficients_x, bound_x_min, bound_x_max = compute_weight_coefficients(
                input_width, output_width, ratio_x, mode, exclude_outside, dev)
            weight_coefficients_y, bound_y_min, bound_y_max = compute_weight_coefficients(
                input_height, output_height, ratio_y, mode, exclude_outside, dev)
            # currently set to uint16
            bound_x_dtype = bound_y_dtype = bits2dtype(16, False)
            # _, bound_x_dtype = range2dtype(0, input_width)
            # _, bound_y_dtype = range2dtype(0, input_height)
            self.constants["bound_x_min"] = PyTensor(
                self.name+"/bound_x_min", bound_x_min.cpu().numpy().astype(dtype2nptype(bound_x_dtype)))
            self.constants["bound_x_max"] = PyTensor(
                self.name+"/bound_x_max", bound_x_max.cpu().numpy().astype(dtype2nptype(bound_x_dtype)))
            self.constants["bound_y_min"] = PyTensor(
                self.name+"/bound_y_min", bound_y_min.cpu().numpy().astype(dtype2nptype(bound_y_dtype)))
            self.constants["bound_y_max"] = PyTensor(
                self.name+"/bound_y_max", bound_y_max.cpu().numpy().astype(dtype2nptype(bound_y_dtype)))

            weight_coefficients_x = (weight_coefficients_x*multiplier).int()
            weight_coefficients_y = (weight_coefficients_y*multiplier).int()
            weight_dtype = bits2dtype(interp_shift, False)
            self.constants["weight_coefficients_x"] = PyTensor(
                self.name+"/weight_coefficients_x", weight_coefficients_x.cpu().numpy().astype(dtype2nptype(weight_dtype)))
            self.constants["weight_coefficients_y"] = PyTensor(
                self.name+"/weight_coefficients_y", weight_coefficients_y.cpu().numpy().astype(dtype2nptype(weight_dtype)))
        else:
            # interp_shift = 13
            interp_shift = extra_params[1]
    self.params['interp_shift_value'] = interp_shift
    self.params['interp_shift_type'] = SHIFT_DTYPE
