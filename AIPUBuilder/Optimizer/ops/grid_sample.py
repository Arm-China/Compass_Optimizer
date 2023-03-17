# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_ERROR
import torch


def PixelAtGrid(feature, n, c, y, x, h, w, resize_height, resize_width, padding_mode, feature_zp):
    # default border
    y_less0_mask = y < 0
    y_greater_mask = y > (h-1)
    x_less0_mask = x < 0
    x_greater_mask = x > (w-1)
    y_c = y.clone()
    x_c = x.clone()
    y_c[y_less0_mask] = 0
    y_c[y_greater_mask] = h-1
    x_c[x_less0_mask] = 0
    x_c[x_greater_mask] = w-1
    out = feature[n, y_c, x_c, :]
    if padding_mode == "border":
        out = out.reshape(resize_height, resize_width, c)
    if padding_mode == "zeros":
        out = out.reshape(resize_height*resize_width, c)
        out[y_less0_mask, :] = -feature_zp
        out[y_greater_mask, :] = -feature_zp
        out[x_less0_mask, :] = -feature_zp
        out[x_greater_mask, :] = -feature_zp
        out = out.reshape(resize_height, resize_width, c)
    return out


def quant_grid_sample(inp0, inp1, method, padding_mode, align_corners, do_scale, shifts):
    feature = inp0.betensor.int()
    grid = inp1.betensor.int()
    feature_batch = feature.shape[0]
    feature_height = feature.shape[1]
    feature_width = feature.shape[2]
    feature_channel = feature.shape[3]
    resize_height = grid.shape[1]
    resize_width = grid.shape[2]
    offset_w = (feature_width-1)
    offset_h = (feature_height-1)

    grid += int(inp1.zerop)
    feature_zp = int(inp0.zerop)
    act_qmin, act_qmax = -2**31, 2**31-1
    shift_0 = shifts[0]
    do_shift = shifts[1]
    shift_diff = do_shift - shift_0

    quant_output = torch.zeros((feature_batch, resize_height, resize_width,
                               feature_channel), device=inp0.betensor.device)
    for n in range(feature_batch):
        qgrid_x = grid[n, :, :, 0].reshape(-1,)
        qgrid_y = grid[n, :, :, 1].reshape(-1,)
        if align_corners:
            q_ix = ((qgrid_x * do_scale) >> shift_0) * offset_w + offset_w * 2 ** (shift_diff)
            q_iy = ((qgrid_y * do_scale) >> shift_0) * offset_h + offset_h * 2 ** (shift_diff)
        else:
            q_ix = ((qgrid_x * do_scale) >> shift_0) * feature_width + \
                offset_w * 2 ** (shift_diff)
            q_iy = ((qgrid_y * do_scale) >> shift_0) * feature_height + \
                offset_h * 2 ** (shift_diff)

        if method == 'nearest':
            q_ix += 2 ** (shift_diff)
            q_iy += 2 ** (shift_diff)
            left_x = (q_ix >> (shift_diff+1)).long()
            top_y = (q_iy >> (shift_diff+1)).long()
            quant_output[n, :, :, :] = PixelAtGrid(feature, n, feature_channel, top_y, left_x,
                                                   feature_height, feature_width, resize_height, resize_width, padding_mode, feature_zp)
        else:
            left_x = (q_ix >> (shift_diff+1)).long()
            top_y = (q_iy >> (shift_diff+1)).long()
            right_x = left_x + 1
            bottom_y = top_y + 1

            x_terp = q_ix - left_x * 2 ** (shift_diff+1)
            y_terp = q_iy - top_y * 2 ** (shift_diff+1)
            x_terp = x_terp.reshape(resize_height, resize_width, 1)
            y_terp = y_terp.reshape(resize_height, resize_width, 1)

            top_left = PixelAtGrid(feature, n, feature_channel, top_y, left_x, feature_height,
                                   feature_width, resize_height, resize_width, padding_mode, feature_zp)
            top_right = PixelAtGrid(feature, n, feature_channel, top_y, right_x, feature_height,
                                    feature_width, resize_height, resize_width, padding_mode, feature_zp)
            bottom_left = PixelAtGrid(feature, n, feature_channel, bottom_y, left_x, feature_height,
                                      feature_width, resize_height, resize_width, padding_mode, feature_zp)
            bottom_right = PixelAtGrid(feature, n, feature_channel, bottom_y, right_x, feature_height,
                                       feature_width, resize_height, resize_width, padding_mode, feature_zp)

            top = top_left + (((top_right - top_left) * x_terp) >> (shift_diff+1))
            bottom = bottom_left + (((bottom_right - bottom_left) * x_terp) >> (shift_diff+1))
            quant_output[n, :, :, :] = (top + (((bottom - top) * y_terp) >> (shift_diff+1)))
    return quant_output


@op_register(OpType.GridSample)
def gridsample(self, *args):
    feature = self.inputs[0]
    grid = self.inputs[1]
    out = self.outputs[0]

    method = self.get_param('method').lower()  # BILINEAR/NEAREST
    padding_mode = self.get_param('padding_mode').lower()  # ZEROS/BORDER/REFLECTION
    align_corners = self.get_param('align_corners')

    if feature.betensor.dim() != 4:
        OPT_ERROR(f"GripSample op now only supports 4-dims feature, now input0 dim is {str(feature.betensor.dim())}.")

    _support_method = ['nearest', 'bilinear']
    if method not in _support_method:
        OPT_WARN(f"GripSample op now only supports {str(_support_method)} method, but now method={method}, "
                 f"and Opt will use 'bilinear' method to continue.")
        method = 'bilinear'

    _support_mode = ['zeros', 'border']
    if padding_mode not in _support_mode:
        OPT_WARN(f"GripSample op now only supports padding_mode('zeros'/'border'), but now padding_mode={padding_mode}, "
                 f"and Opt will use 'zeros' padding_mode to continue.")
        padding_mode = 'zeros'

    if self.quantized:
        shifts = self.get_param('shift_value')
        do_scale = self.get_param('scale_value')
        output = quant_grid_sample(feature,
                                   grid,
                                   method,
                                   padding_mode,
                                   align_corners,
                                   do_scale,
                                   shifts
                                   )
        self.outputs[0].betensor = torch.clamp(output, out.qmin, out.qmax)
    else:
        feature_t = nhwc2nchw(feature.betensor)
        output = torch.nn.functional.grid_sample(
            feature_t.double(), grid.betensor.double(), mode=method, padding_mode=padding_mode, align_corners=align_corners)
        self.outputs[0].betensor = nchw2nhwc(output)

    return self.outputs[0].betensor


@quant_register(OpType.GridSample)
def gridsample_quantize(self, *args):
    import math
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_activation = self.attrs["q_bits_activation"]

    pre_shift = 0

    inp = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    out.qmin, out.qmax = dtype2range(out.dtype)

    total_scale = 1./inp1.scale
    doscale, doscale_type, doshift, doshift_type = get_scale_approximation_params(
        total_scale, 14, force_shift_positive=self.force_shift_positive)

    doscale = int(doscale)
    doshift1 = int(doshift)
    feature_size = max(inp.betensor.shape[1], inp.betensor.shape[2])
    feature_bits = math.ceil(math.log2(feature_size))
    scale_bits = math.ceil(math.log2(doscale))
    inp1_bits = inp1.qbits + (1 if inp1.zerop or not is_signed(inp1.dtype) else 0)
    placehold0_shift_bits = scale_bits + inp1_bits + feature_bits - 30
    placehold1_shift_bits = feature_bits + doshift - 30
    pre_shift = max(0, placehold0_shift_bits, placehold1_shift_bits)

    self.params['scale_value'] = int(doscale)
    self.params['scale_type'] = doscale_type
    self.params['shift_value'] = [int(pre_shift), int(doshift)]
    self.params['shift_type'] = [doshift_type, doshift_type]
