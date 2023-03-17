# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.logger import OPT_ERROR, OPT_WARN
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch
import math
from torchvision.ops import roi_align


def local_float_roi_align(fm, rois, params):
    out_height, out_width = params['output_size']
    spatial_y, spatial_x = params['spatial_scale']
    h_sample_ratio, w_sample_ratio = params['sample_ratio']
    method = params['method'].lower()
    is_half_pixel = params['is_half_pixel']

    # nhwc
    in_height, in_width, in_depth = fm.shape[1:]
    roi_num = rois.shape[0]
    dev = fm.device
    output_shape = [roi_num, out_height, out_width, fm.shape[-1]]
    out = torch.zeros(*output_shape)
    half_pixel_offset = 0.5 if is_half_pixel else 0.
    for r in range(roi_num):
        batch_idx = rois[r, 0]
        if batch_idx < 0:
            OPT_WARN(f"RoiAlign layer: the batch_index of box_id={r} is {batch_idx} < 0.")
            continue
        if batch_idx > fm.shape[0]-1:
            OPT_ERROR(f"RoiAlign layer: batch_index={batch_idx} should be < the featuremap batch_size={fm.shape[0]}")
            continue

        batch_base = torch.flatten(fm[batch_idx.long()])
        w_roi_start = rois[r, 2] * spatial_x - half_pixel_offset
        w_roi_end = rois[r, 4] * spatial_x - half_pixel_offset
        h_roi_start = rois[r, 1] * spatial_y - half_pixel_offset
        h_roi_end = rois[r, 3] * spatial_y - half_pixel_offset

        roi_width = w_roi_end - w_roi_start
        roi_height = h_roi_end - h_roi_start
        if not is_half_pixel:
            roi_width = torch.maximum((roi_width), torch.tensor(1.0, device=dev))
            roi_height = torch.maximum((roi_height), torch.tensor(1.0, device=dev))

        w_step_size = roi_width / out_width
        h_step_size = roi_height / out_height

        # if sampling_ratio=0, use adaptive value of ceil(roi_width/out_width), same for height
        w_sampling_ratio = w_sample_ratio if w_sample_ratio > 0 else int(math.ceil(w_step_size))
        h_sampling_ratio = h_sample_ratio if h_sample_ratio > 0 else int(math.ceil(h_step_size))

        w_bin_size = w_step_size / w_sampling_ratio
        h_bin_size = h_step_size / h_sampling_ratio

        for i in range(out_height):
            for j in range(out_width):
                w_start = w_step_size * j + w_roi_start
                w_end = w_step_size * (j + 1) + w_roi_start
                h_start = h_step_size * i + h_roi_start
                h_end = h_step_size * (i + 1) + h_roi_start

                if method == 'avg':
                    out_k = torch.zeros(in_depth, device=dev)
                else:  # method == max
                    out_k = torch.full([in_depth], torch.finfo(torch.float32).min)
                for y_ind in range(h_sampling_ratio):
                    y = h_start + h_bin_size / 2 + h_bin_size * y_ind
                    for x_ind in range(w_sampling_ratio):

                        x = w_start + w_bin_size / 2 + w_bin_size * x_ind

                        if y < -1.0 or y > in_height or x < -1.0 or x > in_width:
                            ws = [0., 0., 0., 0.]
                            offset = [0, 0, 0, 0]
                        else:
                            y = torch.minimum(torch.maximum(y, torch.tensor(0., device=dev)),
                                              torch.tensor(in_height-1, dtype=torch.float32, device=dev))
                            x = torch.minimum(torch.maximum(x, torch.tensor(0., device=dev)),
                                              torch.tensor(in_width-1, dtype=torch.float32, device=dev))
                            x_low = torch.floor(x)
                            y_low = torch.floor(y)
                            x_high = torch.minimum((x_low + 1), torch.tensor(in_width - 1, device=dev))
                            y_high = torch.minimum((y_low + 1), torch.tensor(in_height - 1, device=dev))
                            dx1 = x - x_low
                            dy1 = y - y_low
                            dx2 = 1. - dx1
                            dy2 = 1. - dy1

                            ws = [dx2 * dy2, dx1 * dy2, dx2 * dy1, dx1 * dy1]
                            offset = [y_low * in_width * in_depth + x_low * in_depth,
                                      y_low * in_width * in_depth + x_high * in_depth,
                                      y_high * in_width * in_depth + x_low * in_depth,
                                      y_high * in_width * in_depth + x_high * in_depth
                                      ]
                            offset = [o.long() for o in offset]
                        for k in range(in_depth):
                            if method == 'avg':
                                interpolation = 0
                                for c in range(4):
                                    interpolation += ws[c] * batch_base[offset[c] + k]
                                out_k[k] += interpolation
                            else:  # max
                                max_4points = torch.max(torch.tensor([ws[0] * batch_base[offset[0] + k],
                                                                      ws[1] * batch_base[offset[1] + k],
                                                                      ws[2] * batch_base[offset[2] + k],
                                                                      ws[3] * batch_base[offset[3] + k],
                                                                      ]))
                                out_k[k] = torch.maximum(out_k[k], max_4points)

                if method == 'avg':
                    out_k = out_k / (w_sampling_ratio * h_sampling_ratio)
                else:  # max
                    pass
                out[r, i, j, :] = out_k
    return out.to(dev)


def quant_roi_align(fm, rois, method, is_half_pixel, pooled_shape, sample, spatial, spatial_shift, scale_shift_pairs, o_qmin, o_qmax):

    do_scale, do_shift = scale_shift_pairs['total_scale_shift']
    out_h_scale, out_h_shift = scale_shift_pairs['out_h_scale_shift']
    out_w_scale, out_w_shift = scale_shift_pairs['out_w_scale_shift']
    sample_h_scale_shift = scale_shift_pairs['sample_h_scale_shift']
    sample_w_scale_shift = scale_shift_pairs['sample_w_scale_shift']
    roi_scale, roi_shift = scale_shift_pairs['roi_scale_shift']

    half_pixel_offset = 2 ** (spatial_shift - 1) if is_half_pixel else 0  # 0.5 * 2 ** spatial_shift
    out_h, out_w = pooled_shape
    sample_h_ratio, sample_w_ratio = sample
    spatial_y, spatial_x = spatial
    roi_num = rois.shape[0]
    _, fm_height, fm_width, fm_channel = fm.shape[:]
    total_shift = do_shift + spatial_shift
    dev = fm.device
    out = torch.zeros(roi_num, out_h, out_w, fm_channel)
    for box_idx in range(roi_num):
        batch_idx = rois[box_idx, 0]
        if batch_idx < 0:
            OPT_WARN(f"RoiAlign layer: the batch_index of box_id={box_idx} is {batch_idx} < 0.")
            continue
        if batch_idx > fm.shape[0]-1:
            OPT_ERROR(f"RoiAlign layer: batch_index={batch_idx} should be < the featuremap batch_size={fm.shape[0]}")
            continue
        # opt impl
        # y_start = ((rois[box_idx, 1] * spatial_y * roi_scale) >> roi_shift).int() - half_pixel_offset
        # x_start = ((rois[box_idx, 2] * spatial_x * roi_scale) >> roi_shift).int() - half_pixel_offset
        # y_end = ((rois[box_idx, 3] * spatial_y * roi_scale) >> roi_shift).int() - half_pixel_offset
        # x_end = ((rois[box_idx, 4] * spatial_x * roi_scale) >> roi_shift).int() - half_pixel_offset
        # lib impl
        y_start = ((rois[box_idx, 1] * roi_scale * 0.5 ** roi_shift).int() * spatial_y) - half_pixel_offset
        x_start = ((rois[box_idx, 2] * roi_scale * 0.5 ** roi_shift).int() * spatial_x) - half_pixel_offset
        y_end = ((rois[box_idx, 3] * roi_scale * 0.5 ** roi_shift).int() * spatial_y) - half_pixel_offset
        x_end = ((rois[box_idx, 4] * roi_scale * 0.5 ** roi_shift).int() * spatial_x) - half_pixel_offset

        data = fm[batch_idx.long()].reshape(-1)

        roi_width = x_end - x_start
        roi_height = y_end - y_start
        if not is_half_pixel:
            roi_width = torch.maximum(roi_width, torch.tensor(2 ** spatial_shift, device=dev))
            roi_height = torch.maximum(roi_height, torch.tensor(2 ** spatial_shift, device=dev))
        step_size_qw = (roi_width * out_w_scale) >> out_w_shift
        step_size_qh = (roi_height * out_h_scale) >> out_h_shift
        # step_size_w = step_size_qw / spatial_shift
        # step_size_h = step_size_qh / spatial_shift

        new_sample_h = sample_h_ratio
        new_sample_w = sample_w_ratio
        sample_h_scale, sample_h_shift = sample_h_scale_shift
        real_sample_h_scale = 1
        real_sample_h_shift = 0
        sample_w_scale, sample_w_shift = sample_w_scale_shift
        real_sample_w_scale = 1
        real_sample_w_shift = 0
        if sample_h_ratio <= 0:
            OPT_ERROR(f"optimizer quant forward now does not support sample_h <=0")
        if sample_w_ratio <= 0:
            OPT_ERROR(f"optimizer quant forward now does not support sample_w <=0")

        # lib impl
        wBinSize = int(step_size_qw * sample_w_scale / 2 ** sample_w_shift)
        hBinSize = int(step_size_qh * sample_h_scale / 2 ** sample_h_shift)
        for i in range(out_h):
            for j in range(out_w):
                h_start = step_size_qh * i + y_start
                w_start = step_size_qw * j + x_start
                if method == 'avg':
                    outdata_batch = [0] * fm_channel
                else:
                    outdata_batch = [-2 ** 31] * fm_channel

                for yInd in range(new_sample_h):
                    for xInd in range(new_sample_w):
                        # y = h_start + (((2 * yInd + 1) * step_size_qh * sample_h_scale) >> sample_h_shift)
                        # x = w_start + (((2 * xInd + 1) * step_size_qw * sample_w_scale) >> sample_w_shift)
                        # lib impl
                        y = h_start + hBinSize * (2 * yInd + 1)
                        x = w_start + wBinSize * (2 * xInd + 1)

                        y1 = y >> spatial_shift
                        y2 = y1 + 1
                        dy1 = y - (y1 << spatial_shift)
                        x1 = x >> spatial_shift
                        x2 = x1 + 1
                        dx1 = x - (x1 << spatial_shift)

                        if y1 < -1 or y1 > fm_height or x1 < -1 or x1 > fm_width:
                            ws = [0, 0, 0, 0]
                            offsets = [0, 0, 0, 0]
                        else:
                            y1 = min(max(y1, 0), fm_height - 1)
                            x1 = min(max(x1, 0), fm_width - 1)

                            if x1 >= fm_width - 1:
                                x1 = x2 = torch.tensor(fm_width - 1, device=fm.device)
                                dx1 = torch.tensor(0, device=fm.device)
                            dx2 = 2 ** spatial_shift - dx1

                            if y1 >= fm_height - 1:
                                y1 = y2 = torch.tensor(fm_height - 1, device=fm.device)
                                dy1 = torch.tensor(0, device=fm.device)
                            dy2 = 2 ** spatial_shift - dy1

                            ws = (dx2 * dy2, dx1 * dy2, dx2 * dy1, dx1 * dy1)
                            ws = [wss >> spatial_shift for wss in ws]
                            offsets = (y1 * fm_width * fm_channel + x1 * fm_channel,
                                       y1 * fm_width * fm_channel + x2 * fm_channel,
                                       y2 * fm_width * fm_channel + x1 * fm_channel,
                                       y2 * fm_width * fm_channel + x2 * fm_channel)
                        for kk in range(fm_channel):
                            if method == 'avg':
                                interpolation = 0
                                for num in range(0, 4):
                                    interpolation += ws[num] * data[offsets[num] + kk]
                                outdata_batch[kk] += interpolation
                            else:  # max
                                max_4points = max([ws[0] * data[offsets[0] + kk],
                                                   ws[1] * data[offsets[1] + kk],
                                                   ws[2] * data[offsets[2] + kk],
                                                   ws[3] * data[offsets[3] + kk],
                                                   ])
                                max_4points = (max_4points * do_scale / 2 ** total_shift)
                                outdata_batch[kk] = max([outdata_batch[kk], max_4points])
                if method == 'avg':
                    '''
                    outdata =  outdata / (sample_h * sample_w)
                    if sample_h and sample_w all > 0:
                        1/(sample_h*sample_w) has included in do_scale/do_shift
                    elif sample_h <= 0 and sample_w > 0:
                        1. /sample_h is from the lut and now use the real_sample_h_scale/real_sample_h_shift. 1./sample_w has included in do_scale/do_shift
                    elif sample_w <= 0 and sample_h > 0:
                        1. /sample_w is from the lut and now use the real_sample_w_scale/real_sample_w_shift. 1./sample_h has included in do_scale/do_shift
                    else: # sample_w <=0 and sample_h <= 0:
                        all 1./sample_w and 1./sample_h are from the lut, so we use the real_sample_h_scale/real_sample_h_shift, real_sample_w_scale/real_sample_w_shfit
                    '''
                    all_shift = do_shift + spatial_shift + real_sample_h_shift + real_sample_w_shift
                    all_scale = do_scale * real_sample_h_scale * real_sample_w_scale
                    for kk in range(fm_channel):
                        outdata_batch[kk] = torch.round((outdata_batch[kk] * all_scale) * 0.5 ** all_shift)
                else:  # method == 'max'
                    pass

                # out[box_idx, i, j, :] = torch.clamp(torch.tensor(outdata_batch), o_qmin, o_qmax)
                out[box_idx, i, j, :] = torch.tensor(outdata_batch)
    return out.to(fm.device)


@op_register(OpType.RoiAlign)
def roialign(self, *args):

    sample_h,  sample_w = self.get_param('sample')
    pool_height, pool_width = self.get_param('pooled_shape')
    method = self.get_param('method').lower()
    spatial_y, spatial_x = self.get_param('spatial_scale_value')
    coordinate_transformation_mode = self.get_param('coordinate_transformation_mode')

    _support_mode = ['avg', 'max']
    if method not in _support_mode:
        OPT_WARN(
            f"Now RoiAlign op supports {str(_support_mode)}, but now is {method}, and we will use the 'avg' to run.")

    _support_coord_trans_mode = ['HALF_PIXEL', 'OUTPUT_HALF_PIXEL']
    if coordinate_transformation_mode not in _support_coord_trans_mode:
        OPT_WARN(
            f"Now RoiAlign op supports {str(_support_coord_trans_mode)}, but now is {coordinate_transformation_mode}.")
    is_half_pixel = coordinate_transformation_mode == 'HALF_PIXEL'
    inp_d = self.inputs[0].betensor
    rois = self.inputs[1].betensor

    if self.quantized:
        o_qmin, o_qmax = self.outputs[0].qmin, self.outputs[0].qmax
        spatial_shift = self.get_param('spatial_shift_value')
        do_scale, inp_do_scale = self.get_param("scale_value")
        do_shift, inp_do_shift = self.get_param("shift_value")
        out_h_do_scale, out_w_do_scale = self.get_param('bin_scale_value')
        out_h_do_shift, out_w_do_shift = self.get_param('bin_shift_value')
        sample_h_scale, sample_w_scale = self.get_param('grid_scale_value')
        sample_h_shift, sample_w_shift = self.get_param('grid_shift_value')

        scale_shift_pairs = {}
        scale_shift_pairs['total_scale_shift'] = (do_scale, do_shift)
        scale_shift_pairs['out_h_scale_shift'] = (out_h_do_scale, out_h_do_shift)
        scale_shift_pairs['out_w_scale_shift'] = (out_w_do_scale, out_w_do_shift)
        scale_shift_pairs['roi_scale_shift'] = (inp_do_scale, inp_do_shift)

        scale_shift_pairs['sample_h_scale_shift'] = (sample_h_scale, sample_h_shift)
        scale_shift_pairs['sample_w_scale_shift'] = (sample_w_scale, sample_w_shift)

        inp_d_zp = inp_d + self.inputs[0].zerop
        rois_zp = rois + self.inputs[1].zerop
        out = quant_roi_align(inp_d_zp,
                              rois_zp,
                              method,
                              is_half_pixel,
                              [pool_height, pool_width],
                              [sample_h, sample_w],
                              [spatial_y, spatial_x],
                              spatial_shift,
                              scale_shift_pairs,
                              o_qmin,
                              o_qmax
                              )
        out = linear_quantize_clip(out, 1.0, self.outputs[0].zerop, o_qmin, o_qmax)
    else:
        if method == 'avg':
            if spatial_x != spatial_y or sample_w != sample_h or coordinate_transformation_mode == 'HALF_PIXEL':
                params = {}
                params['output_size'] = self.get_param('pooled_shape')
                params['sample_ratio'] = self.get_param('sample')
                params['spatial_scale'] = self.get_param('spatial_scale_value')
                params['method'] = method
                params['is_half_pixel'] = is_half_pixel
                out = local_float_roi_align(inp_d, rois, params)
            else:
                split_rois = torch.split(rois, split_size_or_sections=1, dim=-1)
                rois = torch.cat([split_rois[0], split_rois[2], split_rois[1], split_rois[4], split_rois[3]], dim=-1)
                fm = nhwc2nchw(inp_d)
                out = roi_align(fm, rois, [pool_height, pool_width], spatial_scale=spatial_x, sampling_ratio=sample_w)
                out = nchw2nhwc(out)
        else:  # method=='max'
            params = {}
            params['output_size'] = self.get_param('pooled_shape')
            params['sample_ratio'] = self.get_param('sample')
            params['spatial_scale'] = self.get_param('spatial_scale_value')
            params['method'] = method
            params['is_half_pixel'] = is_half_pixel
            out = local_float_roi_align(inp_d, rois, params)

    self.outputs[0].betensor = out
    return self.outputs[0].betensor


@quant_register(OpType.RoiAlign)
def roialign_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]
    scaling_bits = self.attrs['scaling_bits']  # [quant_bit=12, spatial_shift=10]
    # quant_bits = 12
    quant_bits = scaling_bits[0]

    inp = self.inputs[0]
    out = self.outputs[0]
    out_h, out_w = self.get_param('pooled_shape')
    sample_h, sample_w = self.get_param('sample')
    fm_h, fm_w = inp.betensor.shape[1:3]

    out.qbits = q_bits_activation
    out_sign = is_signed(inp.dtype)
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, out_sign)
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = inp.scale, inp.zerop, inp.qmin, inp.qmax, inp.dtype
    inp_do_scale, inp_do_scale_type, inp_do_shift, inp_do_shift_type = \
        get_scale_approximation_params(1. / self.inputs[1].scale,
                                       mult_bits=self.inputs[1].qbits,
                                       force_shift_positive=self.force_shift_positive)

    quant_bits = max(quant_bits, out.qbits)
    out_h_do_scale, out_h_do_scale_type, out_h_do_shift, out_h_do_shift_type = \
        get_scale_approximation_params(1. / out_h,
                                       mult_bits=quant_bits,
                                       force_shift_positive=self.force_shift_positive)
    out_w_do_scale, out_w_do_scale_type, out_w_do_shift, out_w_do_shift_type = \
        get_scale_approximation_params(1. / out_w,
                                       mult_bits=quant_bits,
                                       force_shift_positive=self.force_shift_positive)

    def handle_sample(sample, out_size, fm_size):
        s_do_scale, s_do_shift = 1., 0
        if sample > 0:
            s_do_scale, s_do_scale_type, s_do_shift, s_do_shift_type = \
                get_scale_approximation_params(1. / (2 * sample), mult_bits=quant_bits,
                                               force_shift_positive=self.force_shift_positive)
        return s_do_scale, s_do_scale_type, s_do_shift, s_do_shift_type

    sample_h_do_scale, sample_h_do_scale_type, sample_h_do_shift, sample_h_do_shift_type = handle_sample(
        sample_h, out_h, fm_h)
    sample_w_do_scale, sample_w_do_scale_type, sample_w_do_shift, sample_w_do_shift_type = handle_sample(
        sample_w, out_w, fm_w)

    if sample_h <= 0 and sample_w <= 0:
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(out.scale / inp.scale,
                                           mult_bits=out.qbits,
                                           force_shift_positive=self.force_shift_positive)
    elif sample_h == 0:
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(out.scale / (inp.scale * sample_w),
                                           mult_bits=out.qbits,
                                           force_shift_positive=self.force_shift_positive)
    elif sample_w == 0:
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(out.scale / (inp.scale * sample_h),
                                           mult_bits=out.qbits,
                                           force_shift_positive=self.force_shift_positive)
    else:
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(out.scale / (inp.scale * sample_h * sample_w),
                                           mult_bits=out.qbits,
                                           force_shift_positive=self.force_shift_positive)

    self.params["shift_value"] = [int(do_shift), int(inp_do_shift)]
    self.params["shift_type"] = [do_shift_type, inp_do_shift_type]
    self.params["scale_value"] = [int(do_scale), int(inp_do_scale)]
    self.params["scale_type"] = [do_scale_type, inp_do_scale_type]

    self.params['bin_scale_value'] = [int(out_h_do_scale), int(out_w_do_scale)]
    self.params['bin_scale_type'] = [Dtype.UINT16] * 2
    self.params['bin_shift_value'] = [int(out_h_do_shift), int(out_w_do_shift)]
    self.params['bin_shift_type'] = [out_h_do_shift_type, out_w_do_shift_type]

    self.params['grid_scale_value'] = [int(sample_h_do_scale), int(sample_w_do_scale)]
    self.params['grid_scale_type'] = [Dtype.UINT16] * 2
    self.params['grid_shift_value'] = [int(sample_h_do_shift), int(sample_w_do_shift)]
    self.params['grid_shift_type'] = [sample_h_do_shift_type, sample_w_do_shift_type]
    out.qinvariant = inp.qinvariant

    # spatial_shift = 10
    spatial_shift = scaling_bits[1]
    spatial_scale = 2 ** spatial_shift
    spatial_y, spatial_x = self.get_param('spatial_scale_value')
    spatial_x_scale = torch.round(torch.tensor(spatial_x * spatial_scale)).int()
    spatial_y_scale = torch.round(torch.tensor(spatial_y * spatial_scale)).int()
    self.params['spatial_scale_value'] = [spatial_y_scale.item(), spatial_x_scale.item()]
    self.params['spatial_scale_type'] = [Dtype.UINT16, Dtype.UINT16]
    self.params['spatial_shift_value'] = spatial_shift
    self.params['spatial_shift_type'] = SHIFT_DTYPE
