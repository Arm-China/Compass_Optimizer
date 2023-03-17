# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *


# Layer IR,such as
# layer_name=roi_pool5
# layer_type=MaxRoiPool
# layer_bottom=[conv5_3,proposal_reshape]
# layer_bottom_shape=[[1,38,38,512],[300,5]]
# layer_bottom_type=[float32,float32]
# layer_top=[roi_pool5]
# layer_top_shape=[[300,7,7,512]]
# layer_top_type=[float32]
# spatial_scale_value=[0.0625,0.0625]
####################################################################
# https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/layers/roi_pooling_layer.cpp
# currently, this OP is used in caffe fasterrcnn


@op_register(OpType.MaxRoiPool)
def maxroipooling(self, *args):
    out = self.outputs[0].betensor
    feature = self.inputs[0].betensor
    rois_box = self.inputs[1].betensor

    resize_height = self.outputs[0].ir_shape[1]
    resize_width = self.outputs[0].ir_shape[2]
    channel = self.outputs[0].ir_shape[3]
    height, width = feature.shape[1:3]
    # to match some IR shape,no batch dimension
    if rois_box.ndim == 2:
        rois_box = rois_box.unsqueeze(0)
    rois = rois_box[..., 1:]  # [:,0] stand for batch index

    if feature.shape[0] != rois_box.shape[0]:
        OPT_WARN("MaxRoiPool batch dim is not correct")
    spatial_scale_value = [self.get_param('spatial')[0], self.get_param('spatial')[1]]  # [spatial_y, spatial_x] in IR
    dev = feature.device
    if not self.quantized:
        top_data = torch.zeros((rois.shape[0], rois.shape[1], resize_height, resize_width, channel), device=dev)
        for i in range(rois_box.shape[0]):
            batch_index = rois_box[i, :, 0]
            for boxidx in range(rois.shape[1]):
                batch_idx = batch_index[boxidx].int()
                roi_start_w = (torch.floor(rois[i, boxidx, 1] * spatial_scale_value[1] + 0.5)).int()
                roi_end_w = (torch.floor(rois[i, boxidx, 3] * spatial_scale_value[1] + 0.5)).int()
                roi_start_h = (torch.floor(rois[i, boxidx, 0] * spatial_scale_value[0] + 0.5)).int()
                roi_end_h = (torch.floor(rois[i, boxidx, 2] * spatial_scale_value[0] + 0.5)).int()

                roi_height = max(roi_end_h - roi_start_h + 1, 1)
                roi_width = max(roi_end_w - roi_start_w + 1, 1)
                bin_size_h = roi_height / resize_height
                bin_size_w = roi_width / resize_width

                hstart = torch.floor((torch.arange(0, resize_height, device=out.device)) *
                                     bin_size_h)  # have pooled size h,such as case, vector len is 7
                wstart = torch.floor((torch.arange(0, resize_width, device=out.device)) * bin_size_w)
                hend = torch.ceil((torch.arange(1, resize_height + 1, device=out.device)) * bin_size_h)
                wend = torch.ceil((torch.arange(1, resize_width + 1, device=out.device)) * bin_size_w)

                # have pooled size h,such as case, vector len is 7
                hstart = (torch.clamp(hstart + roi_start_h, 0, height)).int()
                hend = (torch.clamp(hend + roi_start_h, 0, height)).int()
                wstart = (torch.clamp(wstart + roi_start_w, 0, width)).int()
                wend = (torch.clamp(wend + roi_start_w, 0, width)).int()

                for ph in range(resize_height):
                    for pw in range(resize_width):
                        if hstart[ph] >= hend[ph] or wstart[pw] >= wend[pw]:
                            continue
                        else:
                            top_data[i, boxidx, ph, pw, :] = torch.max(
                                torch.max(feature[batch_idx, hstart[ph]:hend[ph], wstart[pw]:wend[pw], :], 0)[0], 0)[0]
    else:
        top_data = torch.zeros((rois.shape[0], rois.shape[1], resize_height, resize_width, channel), device=dev)
        # here is unreasonable, but lib remove a parameter which indicate box scale, lib maybe match specific case
        # but opt need match general cases test, so workaround like this
        spatial_scale_int16 = [spatial_scale_value[0], spatial_scale_value[1]]
        half_value = 1 << 15
        rois = rois.long()
        for i in range(rois_box.shape[0]):
            batch_index = rois_box[i, :, 0]
            for boxidx in range(rois.shape[1]):
                batch_idx = batch_index[boxidx].int()
                # rois is origin size, feature map size is reduced by 1/16, origin size=224, feature size=224/16/=14
                expand_start_w = rois[i, boxidx, 1] * spatial_scale_int16[1]
                expand_end_w = rois[i, boxidx, 3] * spatial_scale_int16[1]
                expand_start_h = rois[i, boxidx, 0] * spatial_scale_int16[0]
                expand_end_h = rois[i, boxidx, 2] * spatial_scale_int16[0]

                roi_start_w_q = (expand_start_w // 65536).int().item()
                roi_end_w_q = (expand_end_w // 65536).int().item()
                roi_start_h_q = (expand_start_h // 65536).int().item()
                roi_end_h_q = (expand_end_h // 65536).int().item()

                roi_start_w_q += (1 if (expand_start_w % 65536) >= half_value else 0)
                roi_end_w_q += (1 if (expand_end_w % 65536) >= half_value else 0)
                roi_start_h_q += (1 if (expand_start_h % 65536) >= half_value else 0)
                roi_end_h_q += (1 if (expand_end_h % 65536) >= half_value else 0)

                roi_height_q = max(roi_end_h_q - roi_start_h_q + 1, 1)
                roi_width_q = max(roi_end_w_q - roi_start_w_q + 1, 1)
                bin_size_h_q = (roi_height_q // resize_height)
                bin_size_w_q = (roi_width_q // resize_width)
                bin_size_h_q_mod = roi_height_q % resize_height
                bin_size_w_q_mod = roi_width_q % resize_width

                # have pooled size h,such as case, vector len is 7
                hstart_q = ((torch.arange(0, resize_height)) * bin_size_h_q) + \
                    (bin_size_h_q_mod * torch.arange(0, resize_height)) // resize_height  # >> 8
                wstart_q = ((torch.arange(0, resize_width)) * bin_size_w_q) + \
                    (bin_size_w_q_mod * torch.arange(0, resize_width)) // resize_width  # >> 8

                hend_q = ((torch.arange(1, resize_height+1)) * bin_size_h_q) + \
                    (bin_size_h_q_mod * torch.arange(1, resize_height+1)) // resize_height
                mask = ((bin_size_h_q_mod * torch.arange(1, resize_height+1)) % resize_height) > 0
                hend_q[mask] += 1

                wend_q = ((torch.arange(1, resize_width+1)) * bin_size_w_q) + \
                    (bin_size_w_q_mod * torch.arange(1, resize_width+1)) // resize_width
                mask = ((bin_size_w_q_mod * torch.arange(1, resize_width+1)) % resize_width) > 0
                wend_q[mask] += 1

                hstart_q = torch.clamp(hstart_q + roi_start_h_q, 0, height)
                hend_q = torch.clamp(hend_q + roi_start_h_q, 0, height)
                wstart_q = torch.clamp(wstart_q + roi_start_w_q, 0, width)
                wend_q = torch.clamp(wend_q + roi_start_w_q, 0, width)
                for ph in range(resize_height):
                    for pw in range(resize_width):
                        if hstart_q[ph] >= hend_q[ph] or wstart_q[pw] >= wend_q[pw]:
                            continue
                        else:
                            top_data[i, boxidx, ph, pw, :] = torch.max(
                                torch.max(feature[batch_idx, hstart_q[ph]:hend_q[ph], wstart_q[pw]:wend_q[pw], :], 0)[
                                    0], 0)[0]

    self.outputs[0].betensor = top_data.reshape(self.outputs[0].ir_shape)
    return top_data


@quant_register(OpType.MaxRoiPool)
# layer_bottom_shape=[1,38,38,1024],[1,100,4]
# layer_bottom_type=float32,float32
# layer_top=CropAndResize
# layer_top_shape=[100,14,14,1024]
def maxroipooling_quantize(self, *args):
    inp = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    spatial_scale_x = self.get_param('spatial')[1]  # [1/16, 1/16]
    spatial_scale_y = self.get_param('spatial')[0]  # [1/16, 1/16]
    input1_scale = 1 / inp1.scale
    self.params['spatial'] = [(torch.round(torch.tensor(spatial_scale_y * input1_scale) * 65536)).int().item(),
                              (torch.round(torch.tensor(spatial_scale_x * input1_scale) * 65536)).int().item()]

    # now not use in opt forward and lib.
    # resize_height, resize_width = out.betensor.shape[1:3]
    # self.params['index_scale'] = int(round(32768.0 / max([resize_width, resize_height])))
