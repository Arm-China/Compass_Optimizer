# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.plugins.aipubt_metric_mAP import BasemAPMetric
import torch
import numpy as np
from torch import nn
from torchvision.ops import nms
from collections import defaultdict


@register_plugin(PluginType.Metric, '1.0')
class CenterNetMetric(BasemAPMetric):
    """
    This CenterNetMetric is used for the metric of centernet model in Optimizer.
    The input image size of centerface model is 512x512.
    """

    def __init__(self):

        self.image_size = [512, 512]

        self.confidence = 0.3
        self.num_iou = 0.3
        self.nms = True
        self.cuda = torch.cuda.is_available()

        self.predicts = defaultdict()
        self.targets = defaultdict()
        self.mAP = 0

    def reset(self):
        self.predicts = defaultdict()
        self.targets = defaultdict()
        self.mAP = 0

    def compute(self):
        self.eval_mAP(self.predicts, self.targets, 90, 0)
        return self.mAP

    def report(self):
        return "CenterNet mAP accuracy is %f" % (self.compute())

    def __call__(self, pred, target):
        pred_whs, pred_offset, pred_hms = pred
        outputs = self.decode_bounding_box(pred_hms, pred_whs, pred_offset, self.confidence)
        results = self.post_process(outputs, self.nms, self.num_iou)

        batch = pred[0].shape[0]
        for b in range(batch):
            predict = {}
            if results[b] is None:
                continue
            top_label = results[b][:, 5].cpu().numpy()
            top_conf = results[b][:, 4].cpu().numpy()
            top_boxes = results[b][:, :4].cpu().numpy()
            predict.update({'label_index': top_label})
            predict.update({'bbox': top_boxes})
            predict.update({'confidence': top_conf})
            predict.update({'image_name': target['image_name'][b].cpu().numpy()})

            targets = {}
            for k, v in target.items():
                targets.update({k: v[b].cpu().numpy()})
            BasemAPMetric.extract_obj_all_class(predict, self.predicts)
            BasemAPMetric.extract_obj_all_class(targets, self.targets)

    def decode_bounding_box(self, prediction_hms, prediction_whs, prediction_offsets, confidence):
        def _pool_nms(heat, kernel=3):
            hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1)//2)
            keep = (hmax == heat).float()
            return heat * keep

        prediction_hms = _pool_nms(prediction_hms)
        batches, output_h, output_w, c = prediction_hms.shape

        height_v, width_v = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        width_v, height_v = width_v.flatten().float(), height_v.flatten().float()
        height_v = height_v.to(device=prediction_hms.device)
        width_v = width_v.to(device=prediction_hms.device)

        detects = []
        for batch in range(batches):
            heat_map = prediction_hms[batch].reshape([-1, c])
            prediction_wh = prediction_whs[batch].reshape([-1, 2])
            prediction_offset = prediction_offsets[batch].reshape([-1, 2])

            class_conf, class_pred = torch.max(heat_map, dim=-1)
            mask = class_conf > confidence

            prediction_wh_mask = prediction_wh[mask]
            prediction_offset_mask = prediction_offset[mask]
            if len(prediction_wh_mask) == 0:
                detects.append([])
                continue

            height_v_mask = torch.unsqueeze(height_v[mask] + prediction_offset_mask[..., 1], -1)
            width_v_mask = torch.unsqueeze(width_v[mask] + prediction_offset_mask[..., 0], -1)

            half_w, half_h = prediction_wh_mask[..., 0:1] / 2, prediction_wh_mask[..., 1:2] / 2

            bboxes = torch.cat([width_v_mask - half_w, height_v_mask - half_h,
                               width_v_mask + half_w, height_v_mask + half_h], dim=1)
            bboxes[:, [0, 2]] /= output_w
            bboxes[:, [1, 3]] /= output_h
            detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask], -1),
                                torch.unsqueeze(class_pred[mask], -1).float()], dim=-1)
            detects.append(detect)

        return detects

    def post_process(self, prediction, need_nms, nms_thres=0.4):
        output = [None for _ in range(len(prediction))]

        for i, image_pred in enumerate(prediction):
            detections = prediction[i]
            if len(detections) == 0:
                continue
            unique_labels = detections[:, -1].cpu().unique()

            if detections.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()
            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]
                keep = nms(detections_class[:, :4], detections_class[:, 4], nms_thres)
                max_detections = detections_class[keep]
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
        return output
