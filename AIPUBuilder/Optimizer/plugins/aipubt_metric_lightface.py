# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch
from itertools import product as product
import numpy as np
from AIPUBuilder.Optimizer.plugins.aipubt_metric_widerface import WiderFaceMetric


@register_plugin(PluginType.Metric, '1.0')
class lightfaceMetric(WiderFaceMetric):
    """
    This lightfaceMetric is used for the metric of lightface model in Optimizer.
    The input image size of centerface model is 240x320.
    """

    def __init__(self):

        self.image_size = [240, 320]
        self.scale = torch.Tensor([1, 1, 1, 1])

        self.nms_threshold = 0.3
        self.box_pred = []
        self.box_gt = []
        self.conf_thresh = 0.65
        self.iou_thresh = 0.45

    def __call__(self, pred, target):
        batch_size = pred[0].shape[0]

        for batch in range(batch_size):
            confidences, det_box = pred[0][batch:batch +
                                           1].cpu().numpy(), pred[1][batch:batch+1].cpu().numpy()

            keep_box, _, score = self.fliter_box_with_nms(
                self.image_size[1], self.image_size[0], confidences, det_box)
            dets = []
            if len(keep_box) > 0:
                dets = np.concatenate([keep_box, score.reshape(-1, 1)], axis=1)
            boxes = []
            for box in target['bbox'][batch].cpu().numpy():
                boxes.append(box)
            self.box_pred.append(dets)
            self.box_gt.append([boxes, target['easy'][batch].cpu().numpy(),
                                target['medium'][batch].cpu().numpy(), target['hard'][batch].cpu().numpy()])
    #################################################################################################
    # filter some boxes which confidence is lower than thresh and iou is low(with nms)

    def fliter_box_with_nms(self, width, height, confidences, boxes, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, 2):
            probs = confidences[:, class_index]
            mask = probs > self.conf_thresh
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]

            box_with_probs = np.concatenate(
                [subset_boxes, probs.reshape(-1, 1)], axis=1)
            keep = self.nms(box_with_probs,
                            top_k=top_k,
                            )
            box_with_probs = box_with_probs[keep, :]
            picked_box_probs.append(box_with_probs)
            picked_labels.extend([class_index] * box_with_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def reset(self):
        pass

    def compute(self):
        return self.eval_map(self.box_pred, self.box_gt)

    def report(self):
        aps = self.compute()
        return ("mAP accuracy easy is %f medium is %f hard is %f" % (aps[0], aps[1], aps[2]))
