# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.ops.pad import pad
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import OPT_DEBUG, OPT_INFO
import math


@op_register(OpType.PostNMS2)
def PostNms2(self, *args):
    # input 6 tensors
    # output 3 tensors, but only tensor 0 are useful
    out = self.outputs[:]
    # process parameters
    proposal_cnt = self.get_param('proposal_cnt')
    # get the height and width of the input image.
    height = self.get_param('image_height')
    width = self.get_param('image_width')

    # get bottom node
    bottom = []
    for i, inp in enumerate(self.inputs):
        bottom.append(inp.betensor)
    detection_boxes, box_num_perClass, nms_box_num_perClass, total_class, class_label, keep = bottom[:]

    max_detection_num = detection_boxes.shape[1]
    batch_num = detection_boxes.shape[0]
    keep = keep.int()
    total_class = total_class.int()
    nms_box_num_perClass = nms_box_num_perClass.int()
    max_nms_class_num = nms_box_num_perClass.shape[1]
    max_class_num = class_label.shape[1]

    out_proposal_box = torch.zeros((batch_num, proposal_cnt, 4))

    out_class_score = torch.zeros([1, proposal_cnt])
    total_class_num = total_class[0, 0]

    max_nms_box_num = keep.shape[1]

    # if self.quantized:
    #     #need?
    #     detection_boxes[0, :, 0] = detection_boxes[0, :, 0] * nor_box_scale_h
    #     detection_boxes[0, :, 2] = detection_boxes[0, :, 2] * nor_box_scale_h
    #     detection_boxes[0, :, 1] = detection_boxes[0, :, 1] * nor_box_scale_w
    #     detection_boxes[0, :, 3] = detection_boxes[0, :, 2] * nor_box_scale_w

    box_num = 0
    cur_class_offset_in_proposals = 0
    for class_idx in range(total_class_num):  # detection class 81,91
        for keep_idx in range(max_nms_box_num):  # all classes boxes
            for box_per_class_idx in range(box_num_perClass[0, class_idx]):
                if cur_class_offset_in_proposals + box_per_class_idx >= max_nms_box_num:
                    break
                box_idx_perClass = (keep[0, cur_class_offset_in_proposals + box_per_class_idx])
                if box_num >= proposal_cnt:
                    break
                out_proposal_box[0, box_num, :] = detection_boxes[0,
                                                                  cur_class_offset_in_proposals + box_idx_perClass, :]
                box_num += 1

        cur_class_offset_in_proposals = cur_class_offset_in_proposals + box_num_perClass[0, class_idx]

    box_num = torch.tensor(cur_class_offset_in_proposals, device=out[2].betensor.device)

    out[0].betensor = (out_proposal_box)
    out[1].betensor = (out_class_score)
    out[2].betensor = (box_num)
    return out


@quant_register(OpType.PostNMS2)
def PostNms2_quantize(self, *args):
    # input 6 tensors
    # output 3 tensors, but only tensor 0 are useful
    inp = self.inputs[:]
    out = self.outputs[:]
    out[0].dtype = inp[0].dtype
    out[0].qbits = inp[0].qbits
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    # int IR params
