# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch
import numpy as np
from AIPUBuilder.Optimizer.plugins.aipubt_metric_widerface import WiderFaceMetric


@register_plugin(PluginType.Metric, '1.0')
class CenterFaceMetric(WiderFaceMetric):
    """
    This centerfaceMetric is used for the metric of centerface model in Optimizer.
    The input image size of centerface model is 640x640.
    nms_threshold=0.3, iou_threshold=0.5, heatmap_threshold=0.123.
    """

    def __init__(self):

        self.image_size = [640, 640]
        self.scale = torch.Tensor([1, 1, 1, 1])

        self.nms_threshold = 0.3
        self.box_pred = []
        self.box_gt = []
        self.iou_thresh = 0.5
        self.heatmap_thresh = 0.123

    def __call__(self, pred, target):
        batch_size = pred[0].shape[0]

        for batch in range(batch_size):
            heatmap, scale, offset, lms = pred[0][batch:batch+1].cpu().numpy(), pred[1][batch:batch+1].cpu().numpy(),\
                pred[2][batch:batch+1].cpu().numpy(), pred[3][batch:batch +
                                                              1].cpu().numpy()
            dets, lms = self.postprocess(
                heatmap, lms, offset, scale, self.heatmap_thresh)

            boxes = []
            for box in target['bbox'][batch].cpu().numpy():
                boxes.append(box)
            self.box_pred.append(dets)
            self.box_gt.append([boxes,
                                target['easy'][batch].cpu().numpy(),
                                target['medium'][batch].cpu().numpy(),
                                target['hard'][batch].cpu().numpy()])

    def reset(self):
        pass

    def compute(self):
        return self.eval_map(self.box_pred, self.box_gt)

    def report(self):
        aps = self.compute()
        return ("mAP accuracy easy is %f medium is %f hard is %f" % (aps[0], aps[1], aps[2]))
