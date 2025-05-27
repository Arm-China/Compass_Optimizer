# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import numpy as np
from AIPUBuilder.Optimizer.framework import *


@register_plugin(PluginType.Metric, '1.0')
class cocokpMetric(OptBaseMetric):
    """
    This cocokpMetric is a simplified method metric used for single coco person keypoint networks.
    """

    def __init__(self, num_joints=17, iou_thres=0.5):
        self.iou_thres = iou_thres
        self.num_joints = num_joints
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        self.vars = (sigmas * 2) ** 2
        self.ious = []

    def __call__(self, pred, target):
        batch_size = pred[0].shape[0]
        for b in range(batch_size):
            pd = pred[0][b]
            gt = target[b]
            iou = self.computeOks(pd, gt)
            self.ious.append(iou[0] > self.iou_thres)

    def computeOks(self, pds, gt):
        gt_len = len(gt['area'])
        if len(pds) == 0 or gt_len == 0:
            return []
        ious = np.zeros((len(pds), gt_len))

        for j in range(gt_len):
            kp = gt['keypoints'][j].view(self.num_joints, 3)
            area = gt['area'][j]
            image_shape = gt['shape']
            xg = kp[:, 0]
            yg = kp[:, 1]
            vg = kp[:, 2]
            k1 = np.count_nonzero(vg > 0)
            x0, y0, x1, y1 = gt['bbox'][j]
            for i, pd in enumerate(pds):
                xd = pd[:, 1] * image_shape[0]
                yd = pd[:, 0] * image_shape[1]
                if k1 > 0:
                    dx = xd - xg
                    dy = yd - yg
                else:
                    z = np.zeros(self.num_joints)
                    dx = np.max((z, x0-xd), axis=0)+np.max((z, xd-x1), axis=0)
                    dy = np.max((z, y0-yd), axis=0)+np.max((z, yd-y1), axis=0)
                e = (dx**2 + dy**2) / self.vars / (area + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e).numpy()) / e.shape[0]
        return ious

    def report(self):
        return "kp mean acc is %f" % (self.compute())

    def reset(self):
        super().reset()

    def compute(self):
        return np.mean(self.ious)
