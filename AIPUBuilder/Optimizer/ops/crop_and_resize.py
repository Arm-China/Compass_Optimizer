# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_ERROR
import torch

'''
layer_id=3
layer_name=CropAndResize
layer_type=CropAndResize
layer_bottom=[Placeholder_0,Placeholder_1_0,Placeholder_2_0]
layer_bottom_shape=[[5,224,224,3],[10,4],[10]]
layer_bottom_type=[float32,float32,int32]
layer_top=[CropAndResize_0]
layer_top_shape=[[10,200,100,3]]
layer_top_type=[float32]
crop_size=[200,100]
extrapolation_value=0.0
method=BILINEAR
'''


def crop_and_resize(fm, boxes, box_indices, method, crop_size, extrapolation_value):
    batch, fm_height, fm_width, fm_depth = fm.shape
    crop_height, crop_width = crop_size[0], crop_size[1]
    box_size = boxes.shape[0]

    out = torch.zeros([box_size, crop_height, crop_width, fm_depth], device=fm.device)

    for b in range(box_size):
        b_idx = box_indices[b].long()
        y1, x1, y2, x2 = boxes[b, :]

        if b_idx < 0 or b_idx >= batch:
            OPT_ERROR(f"Error: batch_index {b_idx} out of range [0, {batch}).")

        height_scale = (y2 - y1) * (fm_height - 1) / (crop_height - 1) if crop_height > 1 else 0
        width_scale = (x2 - x1) * (fm_width - 1) / (crop_width - 1) if crop_width > 1 else 0

        top_y_index = 0
        bottom_y_index = 0
        y_lerp = 0
        x_lerp = 0
        for y in range(crop_height):
            in_y = y1 * (fm_height - 1) + y * height_scale if crop_height > 1 else 0.5 * (y1 + y2) * (fm_height - 1)

            if in_y < 0 or in_y > fm_height - 1:
                padded_v = torch.full([1, 1, out.shape[2], out.shape[3]], extrapolation_value)
                out[b, y, :, :] = padded_v
                continue

            if method == 'bilinear':
                top_y_index = torch.floor(in_y).long().item()
                bottom_y_index = torch.ceil(in_y).long().item()
                y_lerp = in_y - top_y_index
            else:  # method == 'nearest':
                in_y = torch.floor(in_y+0.5).long().item()

            for x in range(crop_width):
                in_x = x1 * (fm_width - 1) + x * width_scale if crop_width > 1 else 0.5 * (x1 + x2) * (fm_width - 1)
                if in_x < 0 or in_x > fm_width - 1:
                    padded_v = torch.full([1, 1, 1, out.shape[3]], extrapolation_value)
                    out[b, y, x, :] = padded_v
                    continue

                if method == 'bilinear':
                    left_x_index = torch.floor(in_x).long().item()
                    right_x_index = torch.ceil(in_x).long().item()
                    x_lerp = in_x - left_x_index

                    top_left = fm[b_idx, top_y_index, left_x_index, :]
                    top_right = fm[b_idx, top_y_index, right_x_index, :]
                    bottom_left = fm[b_idx, bottom_y_index, left_x_index, :]
                    bottom_right = fm[b_idx, bottom_y_index, right_x_index, :]

                    top = top_left + (top_right - top_left) * x_lerp
                    bottom = bottom_left + (bottom_right - bottom_left) * x_lerp
                    out[b, y, x, :] = top + (bottom - top) * y_lerp
                else:  # method == 'nearest':
                    '''
                    round() will lead to 1 grid mismatch between python and c++, so the out will also mismatch.
                    like when in_y=0.5, after round(), the python result in_y=0, but tensorflow result in_y=1,
                    which calls the c++ round().
                    '''
                    in_x = torch.floor(in_x+0.5).long().item()
                    out[b, y, x, :] = fm[b_idx, in_y, in_x, :]

    return out


def quant_crop_and_resize(fm, boxes, box_indices, method, crop_size, qextrapolation_value, index_scale, index_shift):

    batch, fm_height, fm_width, fm_depth = fm.shape
    crop_height, crop_width = crop_size[0], crop_size[1]
    box_size = boxes.shape[0]
    dev = fm.device

    out = torch.zeros([box_size, crop_height, crop_width, fm_depth], device=fm.device)
    for b in range(box_size):
        box_idx = box_indices[b].long()
        y1, x1, y2, x2 = boxes[b]

        if box_idx < 0 or box_idx >= batch:
            OPT_ERROR(f"Error: batch_index {box_idx} out of range [0, {batch}).")

        qheight_scale = torch.div((y2 - y1) * (fm_height - 1) * 256,  crop_height - 1,
                                  rounding_mode='trunc').int() >> 8 if crop_height > 1 else 0
        qwidth_scale = torch.div((x2 - x1) * (fm_width - 1) * 256, crop_width - 1,
                                 rounding_mode='trunc').int() >> 8 if crop_width > 1 else 0
        # qheight_scale = (((y2 - y1) * 256) // (crop_height - 1)) >> 8 if crop_height > 1 else 0
        # qwidth_scale = (((x2 - x1) * 256) // (crop_width - 1)) >> 8 if crop_width > 1 else 0

        top_y_index = 0
        bottom_y_index = 0
        y_lerp = 0
        for y in range(crop_height):
            in_y = y1 * (fm_height - 1) + y * qheight_scale
            in_y = in_y.to(torch.int64)
            # in_y = y1 + y * qheight_scale
            if in_y < 0 or (in_y * index_scale >> index_shift) > (fm_height - 1):
                padded_v = torch .full([1, 1, out.shape[2], out.shape[3]], qextrapolation_value)
                out[b, y, :, :] = padded_v
                continue

            if method == 'bilinear':
                top_y_index = (in_y * index_scale >> index_shift).long()
                bottom_y_index = top_y_index + 1  # ((in_y * index_scale + 2 ** index_shift) >> index_shift).long()
                y_lerp = in_y - torch.div(top_y_index * 2 ** index_shift, index_scale, rounding_mode='trunc')
            else:  # method == 'nearest':
                # in_y = ((in_y + 2 ** (qvalue-1)) >> qvalue).long().item()
                in_y = ((in_y * index_scale + 2 ** (index_shift - 1)).long() >> index_shift).item()

            for x in range(crop_width):
                in_x = x1 * (fm_width - 1) + x * qwidth_scale
                in_x = in_x.to(torch.int64)
                # in_x = x1 + x * qwidth_scale
                if in_x < 0 or ((in_x * index_scale) >> index_shift) > (fm_width - 1):
                    padded_v = torch.full([1, 1, 1, out.shape[3]], qextrapolation_value)
                    out[b, y, x, :] = padded_v
                    continue

                if method == 'bilinear':
                    left_x_index = (in_x * index_scale >> index_shift).long()
                    right_x_index = left_x_index + 1  # ((in_x * index_scale + 2 ** index_shift) >> index_shift).long()
                    x_lerp = in_x - torch.div(left_x_index * 2 ** index_shift, index_scale, rounding_mode='trunc')

                    top_y_index = torch.clamp(top_y_index, torch.tensor(0, device=dev),
                                              torch.tensor(fm_height - 1, device=dev))
                    bottom_y_index = torch.clamp(bottom_y_index,
                                                 torch.tensor(0, device=dev),
                                                 torch.tensor(fm_height - 1, device=dev))
                    left_x_index = torch.clamp(left_x_index, torch.tensor(0, device=dev),
                                               torch.tensor(fm_width - 1, device=dev))
                    right_x_index = torch.clamp(right_x_index,
                                                torch.tensor(0, device=dev),
                                                torch.tensor(fm_width - 1, device=dev))

                    top_left = fm[box_idx, top_y_index, left_x_index, :]
                    top_right = fm[box_idx, top_y_index, right_x_index, :]
                    bottom_left = fm[box_idx, bottom_y_index, left_x_index, :]
                    bottom_right = fm[box_idx, bottom_y_index, right_x_index, :]

                    top = top_left + (((top_right - top_left) * x_lerp * index_scale).long() >> index_shift)
                    bottom = bottom_left + (((bottom_right - bottom_left) * x_lerp * index_scale).long() >> index_shift)
                    out[b, y, x, :] = (top + (((bottom - top) * y_lerp * index_scale).long() >> index_shift))
                else:  # method == 'nearest':
                    # in_x = ((in_x + 2 ** (qvalue - 1)) >> qvalue).long().item()
                    in_x = ((in_x * index_scale + 2 ** (index_shift - 1)).long() >> index_shift).item()
                    out[b, y, x, :] = fm[box_idx, in_y, in_x, :]
    return out


@op_register(OpType.CropAndResize)
def cropandresize(self, *args):
    feature_map = self.inputs[0].betensor
    boxes = self.inputs[1].betensor
    box_indices = self.inputs[2].betensor

    method = self.get_param('method').lower()
    crop_size = self.get_param('crop_size')
    extrapolation_value = self.get_param('extrapolation_value')

    _support_method = ['nearest', 'bilinear']
    if method not in _support_method:
        OPT_ERROR(f"CropAndResize op now only supports {str(_support_method)}, but now method={method}, "
                  f"and Opt will use 'nearest' method to continue.")
        method = 'nearest'

    if self.quantized:
        index_shift = self.get_param('shift_value')
        index_scale = self.get_param('scale_value')
        self.outputs[0].betensor = quant_crop_and_resize(feature_map,
                                                         boxes,
                                                         box_indices,
                                                         method,
                                                         crop_size,
                                                         extrapolation_value,
                                                         index_scale,
                                                         index_shift
                                                         )
    else:
        self.outputs[0].betensor = crop_and_resize(
            feature_map, boxes, box_indices, method, crop_size, extrapolation_value)

    return self.outputs[0].betensor


@quant_register(OpType.CropAndResize)
def cropandresize_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    out.qmin, out.qmax = dtype2range(out.dtype)

    # # default enlarge 32768
    # enlarge_scale = torch.round(torch.tensor(32768./self.inputs[1].scale)).int().item()
    # self.params['scale_value'] = enlarge_scale
    # self.params['scale_type'] = bits2dtype(16, False)
    # self.params['shift_value'] = 15
    # self.params['shift_type'] = SHIFT_DTYPE

    total_scale = 1./self.inputs[1].scale
    qbits = self.inputs[1].qbits
    if self.inputs[1].qbits > 8:
        qbits = 2
    dscale, scl_type, dshift, sft_type = get_scale_approximation_params(total_scale,
                                                                        qbits,
                                                                        force_shift_positive=self.force_shift_positive)
    # doscale, doscale_type, doshift, doshift_type = get_scale_approximation_params(
    #     total_scale, out.qbits, force_shift_positive=self.force_shift_positive)

    extrapolation_value = self.get_param('extrapolation_value')
    qextrapolation_value = int(extrapolation_value * 2 ** dshift / dscale)

    self.params['scale_value'] = int(dscale)
    self.params['scale_type'] = scl_type
    self.params['shift_value'] = int(dshift)
    self.params['shift_type'] = sft_type
    self.params['extrapolation_value'] = qextrapolation_value
    self.params['is_perf_mode'] = 'false'
