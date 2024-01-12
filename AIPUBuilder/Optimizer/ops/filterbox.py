# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import *
import torch

register_optype('FilterBoxes')


@op_register(OpType.FilterBoxes)
def filterbox(self, *args):
    batch_proposal_boxes = self.inputs[0].betensor
    batch_proposal_scores = self.inputs[1].betensor
    batch_box_num = self.inputs[2].betensor.int()
    batch_label_perclass = self.inputs[3].betensor.int()
    batch_total_num = self.inputs[4].betensor.int()

    scorethreshold = int(self.params['score_threshold']) if self.quantized else float(self.params['score_threshold'])
    min_size_h, min_size_w = self.params['min_size']
    maxboxnum = self.params['maxnum']
    batch_size = batch_proposal_boxes.shape[0]
    max_class_num = batch_label_perclass.shape[1]
    filter_box = torch.zeros([batch_size, maxboxnum, 4], device=batch_proposal_boxes.device)
    filter_score = torch.zeros([batch_size, maxboxnum], device=batch_proposal_boxes.device)
    filter_boxnum_perclass = torch.zeros([batch_size, max_class_num], device=batch_proposal_boxes.device)
    filter_label_perclass = torch.zeros([batch_size, max_class_num], device=batch_proposal_boxes.device)
    filter_total_class_num = torch.zeros([batch_size, 1], device=batch_proposal_boxes.device)

    for b in range(batch_size):
        proposal_box = batch_proposal_boxes[b]
        proposal_scores = batch_proposal_scores[b]
        proposal_box_num = batch_box_num[b]
        label_perclass = batch_label_perclass[b]
        total_class = batch_total_num[b]

        input_offset = 0
        output_offset = 0
        valid_class_id = 0
        exceed = False
        for class_id in range(total_class):
            box_num = proposal_box_num[class_id]
            if box_num == 0:
                continue
            box = proposal_box[input_offset:box_num + input_offset, :]
            box_h = box[..., 2] - box[..., 0]
            box_w = box[..., 3] - box[..., 1]
            box_score = proposal_scores[input_offset:box_num + input_offset].clone()
            score_mask = box_score > scorethreshold
            minsize_mask = torch.bitwise_and(box_h >= min_size_h, box_w >= min_size_w)
            valid_mask = torch.bitwise_and(score_mask, minsize_mask)
            valid_box = box[valid_mask, :]
            valid_box_num = valid_box.shape[0]
            if valid_box_num == 0:
                input_offset += box_num
                continue

            if output_offset + valid_box_num >= maxboxnum:
                valid_box_num = maxboxnum - output_offset
                exceed = True
            filter_box[b, output_offset:output_offset+valid_box_num, :] = valid_box[:valid_box_num, ...]
            filter_score[b, output_offset:output_offset+valid_box_num] = box_score[valid_mask][:valid_box_num]
            filter_boxnum_perclass[b, valid_class_id] = valid_box_num
            filter_label_perclass[b, valid_class_id] = label_perclass[class_id]

            output_offset += valid_box_num
            input_offset += box_num
            valid_class_id += 1

            if exceed:
                break

        filter_total_class_num[b] = valid_class_id

    self.outputs[0].betensor = filter_box
    self.outputs[1].betensor = filter_score
    self.outputs[2].betensor = filter_boxnum_perclass
    self.outputs[3].betensor = filter_label_perclass
    self.outputs[4].betensor = filter_total_class_num

    return [o.betensor for o in self.outputs]


@quant_register(OpType.FilterBoxes)
def filterbox_quantize(self, *args):
    for idx, _ in enumerate(self.inputs):
        self.outputs[idx].dtype = self.inputs[idx].dtype
        self.outputs[idx].scale = self.inputs[idx].scale
        self.outputs[idx].zerop = self.inputs[idx].zerop
        self.outputs[idx].qbits = self.inputs[idx].qbits
        self.outputs[idx].qinvariant = self.inputs[idx].qinvariant

    minsize_h, minsize_w = self.get_param('min_size')
    score_threshold = float(self.get_param('score_threshold'))

    minsize_h = linear_quantize_clip(
        minsize_h, self.inputs[0].scale, self.inputs[0].zerop, self.inputs[0].qmin, self.inputs[0].qmax).int().item()
    minsize_w = linear_quantize_clip(
        minsize_w, self.inputs[0].scale, self.inputs[0].zerop, self.inputs[0].qmin, self.inputs[0].qmax).int().item()

    if score_threshold == float('-inf'):
        score_min, _ = dtype2range(self.outputs[1].dtype)
        score_threshold = score_min - 1
    else:
        score_threshold = linear_quantize_clip(
            score_threshold, self.inputs[1].scale, self.inputs[1].zerop, self.inputs[1].qmin, self.inputs[1].qmax).int().item()

    self.params['min_size'] = [minsize_h, minsize_w]
    self.params['score_threshold'] = int(score_threshold)
