# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *


register_optype('ROIPooling')

# Layer IR,such as
# layer_name=CropAndResize
# layer_type=ROIPooling
# layer_bottom=FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/add,Reshape_5
# layer_bottom_shape=[1,38,38,1024],[1,100,4]
# layer_bottom_type=float32,float32
# layer_top=CropAndResize
# layer_top_shape=[100,14,14,1024]
# layer_top_type=float32
# x_index_scale=390
# x_index_shift=15
# y_index_scale=390
# y_index_shift=15
####################################################################
# in fact, this op is crop and resize
# currently, this OP is for tensorflow fasterrcnn and has been splited into crop_and_resize + pooling


@op_register(OpType.ROIPooling)
def roipooling(self, *args):
    out = self.outputs[0].betensor
    feature = self.inputs[0].betensor
    nor_box = self.inputs[1].betensor

    resize_height_ = out.shape[1]
    resize_width_ = out.shape[2]
    channel_ = out.shape[3]
    feature_height_, feature_width_ = feature.shape[1:3]

    if not self.quantized:
        resize_feature = torch.zeros((nor_box.shape[1], resize_height_, resize_width_, channel_), device=feature.device)
        # this nor_box has normalized to (0,1)
        # for batchidx in range(feature.shape[0]):
        batch_idx = 0
        for j in range(nor_box.shape[1]):
            y0 = nor_box[batch_idx, j, 0]*feature_height_/resize_height_
            y1 = nor_box[batch_idx, j, 2]*feature_height_/resize_height_
            x0 = nor_box[batch_idx, j, 1]*feature_width_/resize_width_
            x1 = nor_box[batch_idx, j, 3]*feature_width_/resize_width_

            x = ((resize_width_ * x0 + torch.arange(0, resize_width_, device=out.device) * (x1+1 - x0))).int()
            y = ((resize_height_ * y0 + torch.arange(0, resize_height_, device=out.device) * (y1+1 - y0))).int()
            y = torch.clamp(y, 0, feature_height_ - 1)
            x = torch.clamp(x, 0, feature_width_ - 1)
            for idxh in range(resize_height_):
                resize_feature[j, idxh, :, :] = feature[batch_idx, y[idxh].item(), x.long(), :]
    else:
        resize_feature = torch.zeros((nor_box.shape[1], resize_height_, resize_width_, channel_))
        x_index_scale_ = self.get_param('x_index_scale')
        y_index_scale_ = self.get_param('y_index_scale')
        x_index_shift_ = self.get_param('x_index_shift')
        y_index_shift_ = self.get_param('y_index_shift')
        # for batchidx in range(feature.shape[0]):
        batch_idx = 0
        nor_box = torch.clamp(torch.round(nor_box * 255.0) >> 15, 0, 255)

        for j in range(nor_box.shape[1]):
            y0 = nor_box[batch_idx, j, 0]
            y1 = nor_box[batch_idx, j, 2]
            x0 = nor_box[batch_idx, j, 1]
            x1 = nor_box[batch_idx, j, 3]

            x = ((resize_width_ * x0 + torch.arange(0, resize_width_, device=out.device)
                 * (x1 - x0 + 1)).int() * (x_index_scale_)) >> x_index_shift_
            y = ((resize_height_ * y0 + torch.arange(0, resize_height_, device=out.device)
                 * (y1 - y0 + 1)).int() * (y_index_scale_)) >> y_index_shift_
            y = torch.clamp(y, 0, feature_height_ - 1)
            x = torch.clamp(x, 0, feature_width_ - 1)
            for idxh in range(resize_height_):
                resize_feature[j, idxh, :, :] = feature[batch_idx, y[idxh].long(), x.long(), :]

    self.outputs[0].betensor = resize_feature
    return resize_feature


@quant_register(OpType.ROIPooling)
# layer_bottom_shape=[1,38,38,1024],[1,100,4]
# layer_bottom_type=float32,float32
# layer_top=CropAndResize
# layer_top_shape=[100,14,14,1024]
def roipooling_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    # index_precision_ = 16 #fixed ?
    extra_params = self.get_attrs('extra_params', optional=True, default_value=[0, 16])
    index_precision_ = extra_params[1]
    resize_height_, resize_width_ = out.betensor.shape[1:3]
    feature_height_, feature_width_ = inp.betensor.shape[1:3]

    cord_scale_ = 256/max(feature_width_, feature_height_)
    feature_scale_ = torch.round(torch.tensor(cord_scale_)).item()

    x_index_scale_ = torch.round(torch.tensor(32768/feature_scale_ / resize_width_)).item()
    y_index_scale_ = torch.round(torch.tensor(32768/feature_scale_ / resize_height_)).item()
    self.params['x_index_scale'] = x_index_scale_
    self.params['y_index_scale'] = y_index_scale_
    self.params['x_index_shift'] = index_precision_ - 1
    self.params['y_index_shift'] = index_precision_ - 1
    self.params['is_perf_mode'] = 'false'
