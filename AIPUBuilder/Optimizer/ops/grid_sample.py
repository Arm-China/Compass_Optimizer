# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

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


def GsDenormalize(grid,  length, align_corners):
    if align_corners:  # [-1, 1] to [0, length - 1]
        return ((grid + 1) / 2. * (length - 1))
    else:  # [-1, 1] to [-0.5, length - 0.5]
        return (((grid + 1) * length - 1) / 2.)


def grid_clamp(x, x_min, x_max, clip_xmin, clip_xmax):
    less_mask = x < x_min  # torch.bitwise_or(x < x_min, x>x_max)
    greater_mask = x > x_max
    x[less_mask] = clip_xmin
    x[greater_mask] = clip_xmax
    return x


def quant_grid_sample(self, inp0, inp1, method, padding_mode, align_corners, do_scale, shifts):
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
    q16_qmin, q16_qmax = -2**15, 2**15-1
    do_scale0, do_scale1 = do_scale
    do_shift0, do_shift1 = shifts
    gridx_shift = self.params['coordinate_x_shift']
    gridy_shift = self.params['coordinate_y_shift']

    if align_corners:
        x_min = 0
        x_max = offset_w * (2**gridx_shift)
        y_min = 0
        y_max = offset_h * (2**gridy_shift)
    else:
        x_min = -2**(gridx_shift-1)
        x_max = feature_width * (2**gridx_shift) - 2**(gridx_shift-1)
        y_min = -2**(gridy_shift-1)
        y_max = feature_height * (2**gridy_shift) - 2**(gridy_shift-1)

    quant_output = torch.zeros((feature_batch, resize_height, resize_width,
                                feature_channel), device=inp0.betensor.device)
    for n in range(feature_batch):
        qgrid_x = grid[n, :, :, 0].reshape(-1,)
        qgrid_y = grid[n, :, :, 1].reshape(-1,)

        q_ix = (qgrid_x * do_scale0 >> do_shift0) + offset_w * (2**(gridx_shift-1))
        q_iy = (qgrid_y * do_scale1 >> do_shift1) + offset_h * (2**(gridy_shift-1))
        q_ix = torch.clamp(q_ix, q16_qmin, q16_qmax)
        q_iy = torch.clamp(q_iy, q16_qmin, q16_qmax)
        if method == 'nearest':
            q_ix += 2 ** (gridx_shift-1)
            q_iy += 2 ** (gridy_shift-1)
            q_ix = torch.clamp(q_ix, q16_qmin, q16_qmax)
            q_iy = torch.clamp(q_iy, q16_qmin, q16_qmax)
            left_x = (q_ix >> gridx_shift).long()
            top_y = (q_iy >> gridy_shift).long()
            quant_output[n, :, :, :] = PixelAtGrid(feature, n, feature_channel, top_y, left_x,
                                                   feature_height, feature_width, resize_height, resize_width, padding_mode, feature_zp)
        else:
            if padding_mode == 'border':
                q_ix = grid_clamp(q_ix, x_min, x_max, 0, offset_w * (2**gridx_shift))
                q_iy = grid_clamp(q_iy, y_min, y_max, 0, offset_h * (2**gridy_shift))
            left_x = (q_ix.long() >> gridx_shift).long()
            top_y = (q_iy.long() >> gridy_shift).long()
            right_x = left_x + 1
            bottom_y = top_y + 1
            x_terp = q_ix - left_x * 2 ** (gridx_shift)
            y_terp = q_iy - top_y * 2 ** (gridy_shift)
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

            top = top_left + (((top_right - top_left) * x_terp).long() >> (gridx_shift))
            bottom = bottom_left + (((bottom_right - bottom_left) * x_terp).long() >> gridx_shift)
            quant_output[n, :, :, :] = (top + (((bottom - top) * y_terp).long() >> gridy_shift))
    return quant_output


def quant_grid_sample_lookup(self, inp0, inp1, method, padding_mode, align_corners, do_scale, shifts):
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
    feature_zp = int(inp0.zerop)

    act_qmin, act_qmax = -2**31, 2**31-1
    luty = self.constants["luty"] .betensor
    lutx = self.constants["lutx"] .betensor
    gridx_shift = self.params['coordinate_x_shift']
    gridy_shift = self.params['coordinate_y_shift']

    quant_output = torch.zeros((feature_batch, resize_height, resize_width,
                                feature_channel), device=inp0.betensor.device)
    for n in range(feature_batch):
        qgrid_x = grid[n, :, :, 0].reshape(-1,)
        qgrid_y = grid[n, :, :, 1].reshape(-1,)

        lut_in_bits = 8
        in_is_signed = True
        out_is_signed = True
        q_ix = lookup_lut_powerof2(qgrid_x, lutx, lut_in_bits, in_is_signed,
                                   dtype2bits(self.constants["lutx"].dtype), out_is_signed)
        q_iy = lookup_lut_powerof2(qgrid_y, luty, lut_in_bits, in_is_signed,
                                   dtype2bits(self.constants["luty"].dtype), out_is_signed)
        if method == 'nearest':
            quant_output[n, :, :, :] = PixelAtGrid(feature, n, feature_channel, q_iy.int(), q_ix.int(),
                                                   feature_height, feature_width, resize_height, resize_width, padding_mode, feature_zp)
        else:
            x_terp_lut = self.constants["x_terp"].betensor
            y_terp_lut = self.constants["y_terp"].betensor
            left_x = q_ix
            top_y = q_iy
            right_x = left_x + 1
            bottom_y = top_y + 1
            x_terp = lookup_lut_powerof2(qgrid_x, x_terp_lut, lut_in_bits, in_is_signed,
                                         dtype2bits(self.constants["x_terp"].dtype), False)
            y_terp = lookup_lut_powerof2(qgrid_y, y_terp_lut, lut_in_bits, in_is_signed,
                                         dtype2bits(self.constants["y_terp"].dtype), False)

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

            top = top_left + (((top_right - top_left) * x_terp) >> (gridx_shift))
            bottom = bottom_left + (((bottom_right - bottom_left) * x_terp) >> gridx_shift)
            quant_output[n, :, :, :] = (top + (((bottom - top) * y_terp) >> gridy_shift))
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
        grid_bits = dtype2bits(self.inputs[1].dtype)
        if grid_bits <= 8:
            # currently 8bit quantization will not be used
            output = quant_grid_sample_lookup(self,
                                              feature,
                                              grid,
                                              method,
                                              padding_mode,
                                              align_corners,
                                              do_scale,
                                              shifts
                                              )
        else:
            output = quant_grid_sample(self,
                                       feature,
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

    self.outputs[0].dtype = self.inputs[0].dtype
    self.outputs[0].scale = self.inputs[0].scale
    self.outputs[0].zerop = self.inputs[0].zerop
    self.outputs[0].qbits = self.inputs[0].qbits
    self.outputs[0].qinvariant = self.inputs[0].qinvariant
    self.outputs[0].qmin, self.outputs[0].qmax = dtype2range(self.outputs[0].dtype)

    method = self.get_param('method').lower()  # BILINEAR/NEAREST
    padding_mode = self.get_param('padding_mode').lower()  # ZEROS/BORDER/REFLECTION
    align_corners = self.get_param('align_corners')
    feature_height = self.inputs[0].ir_shape[1]
    feature_width = self.inputs[0].ir_shape[2]

    _SUPPORTED_METHOD = ['nearest', 'bilinear']
    _SUPPORTED_PADDING_MODE = ['zeros', 'border']
    if method not in _SUPPORTED_METHOD:
        OPT_FATAL('layerid=%s, gridsample  method only support %s, but now method=%s' %
                  (self.attrs['layer_id'], str(_SUPPORTED_METHOD), method))
    if padding_mode not in _SUPPORTED_PADDING_MODE:
        OPT_FATAL('layerid=%s, gridsample  padding_mode only support %s, but now padding_mode=%s' %
                  (self.attrs['layer_id'], str(_SUPPORTED_PADDING_MODE), padding_mode))

    iqmin, iqmax = dtype2range(self.inputs[1].dtype)
    grid_bits = dtype2bits(self.inputs[1].dtype)
    steps = iqmax - iqmin + 1
    grid_f_range = linear_dequantize(torch.linspace(iqmin, iqmax, steps=steps,
                                                    device=self.inputs[1].betensor.device), self.inputs[1].scale, self.inputs[1].zerop)
    gridx_range = GsDenormalize(grid_f_range,  feature_width,  align_corners)
    gridy_range = GsDenormalize(grid_f_range,  feature_height,  align_corners)
    gridx_range_bits,  _ = range2bits(gridx_range.min(), gridx_range.max(), force_int=True)
    gridy_range_bits,  _ = range2bits(gridy_range.min(), gridy_range.max(), force_int=True)
    if gridx_range_bits > 16 or gridy_range_bits > 16:
        OPT_WARN('layer_id=%s,GridSample Op grid coordinates may exceed range -32768,32767],  which causes accuracy problems, Please Check!' %
                 (str(self.attrs['layer_id'])))
    # currently 8bit quantization will not be used
    if grid_bits <= 8:
        if method == 'nearest':
            gridx_range = torch.round(gridx_range)
            gridy_range = torch.round(gridy_range)
        if padding_mode == 'border':
            x_min = -0.5
            x_max = feature_width - 0.5
            y_min = -0.5
            y_max = feature_height - 0.5
            if align_corners:
                x_min = 0.
                x_max = feature_width - 1.
                y_min = 0.
                y_max = feature_height - 1.
            gridx_range = grid_clamp(gridx_range, x_min, x_max, 0, feature_width-1)
            gridy_range = grid_clamp(gridy_range, y_min, y_max, 0, feature_height-1)
        if method.upper() == 'nearest':
            lutx = gridx_range.int()
            luty = gridy_range.int()
        else:  # 'bilinear':
            lutx = torch.floor(gridx_range).int()
            luty = torch.floor(gridy_range).int()
            x_terp = ((gridx_range - lutx).float() * 1024).int()
            y_terp = ((gridy_range - luty).float() * 1024).int()
            self.constants["x_terp"] = PyTensor(
                self.name+"/x_terp", x_terp.cpu().numpy().astype(dtype2nptype(bits2dtype(16, False))))
            self.constants["y_terp"] = PyTensor(
                self.name+"/y_terp", y_terp.cpu().numpy().astype(dtype2nptype(bits2dtype(16, False))))

        self.constants["lutx"] = PyTensor(
            self.name+"/lutx", lutx.cpu().numpy().astype(dtype2nptype(bits2dtype(16, True))))
        self.constants["luty"] = PyTensor(
            self.name+"/luty", luty.cpu().numpy().astype(dtype2nptype(bits2dtype(16, True))))
        self.params['scale_value'] = [int(1), int(1)]
        self.params['scale_type'] = [Dtype.UINT8, Dtype.UINT8]
        self.params['shift_value'] = [int(0), int(0)]
        self.params['shift_type'] = [Dtype.INT8, Dtype.INT8]
        self.params['coordinate_x_shift'] = 10
        self.params['coordinate_y_shift'] = 10

    else:
        mulpliter_gridx = 16 - gridx_range_bits
        mulpliter_gridy = 16 - gridy_range_bits
        if align_corners:
            feature_height = feature_height - 1
            feature_width = feature_width - 1
        gridx_scale = (2**mulpliter_gridx) * feature_width / self.inputs[1].scale / 2
        gridy_scale = (2**mulpliter_gridy) * feature_height / self.inputs[1].scale / 2
        gridx_doscale, gridx_doscale_type, gridx_doshift, gridx_doshift_type = get_scale_approximation_params(
            gridx_scale, 15, force_shift_positive=self.force_shift_positive)
        gridy_doscale, gridy_doscale_type, gridy_doshift, gridy_doshift_type = get_scale_approximation_params(
            gridy_scale, 15, force_shift_positive=self.force_shift_positive)

        self.params['scale_value'] = [int(gridx_doscale), int(gridy_doscale)]
        self.params['scale_type'] = [gridx_doscale_type, gridy_doscale_type]
        self.params['shift_value'] = [int(gridx_doshift), int(gridy_doshift)]
        self.params['shift_type'] = [gridx_doshift_type, gridy_doshift_type]

        self.params['coordinate_x_shift'] = mulpliter_gridx
        self.params['coordinate_y_shift'] = mulpliter_gridy
