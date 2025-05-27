# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import numpy as np
from collections import defaultdict
from scipy.ndimage import gaussian_filter, maximum_filter


@register_plugin(PluginType.Metric, '1.0')
class PCKhMetric(OptBaseMetric):
    """
    This PCKhMetric is used for the metric of stacked_hourglass_tensorflow model in Optimizer.
    """

    def __init__(self, threshold=0.5, hmap_thres=1e-6, bias=0.6):
        self.ret = []
        self.label = defaultdict(list)
        self.threshold = threshold
        self.hmap_thres = hmap_thres
        self.bias = bias

    def __call__(self, pred, target):
        for idx, heatmap in enumerate(pred[1].cpu().numpy()):
            kps = []
            # post process
            scale = target['scale'][idx].cpu().numpy() * 200
            center = target['center'][idx].cpu().numpy()
            for i in range(heatmap.shape[-1]):
                hmap = gaussian_filter(heatmap[:, :, i], sigma=0.5)
                hmap[hmap < self.hmap_thres] = 0
                mask = (hmap == maximum_filter(hmap, footprint=np.ones((3, 3))))
                hmap_nms = hmap * mask

                y, x = np.where(hmap_nms == hmap_nms.max())
                if len(x) > 0 and len(y) > 0:
                    kp = [int(x[0]), int(y[0]), hmap_nms[y[0], x[0]]]
                else:
                    kp = [0, 0, 0]
                data = [
                    [64 / scale, 0, 64 * (-center[0] / scale + 0.5)],
                    [0, 64 / scale, 64 * (-center[1] / scale + 0.5)],
                    [0, 0, 1]
                ]
                data = np.linalg.inv(data)
                kp = np.dot(data, np.transpose([kp[0] - 1, kp[1] - 1, 1])).astype(int)
                kp += 1

                kps.append(kp[:2])
            self.ret.append(kps)

        for key in target:
            self.label[key].append(target[key].cpu().numpy())

    def reset(self):
        self.ret = []
        self.label = defaultdict(list)

    def compute(self):
        label = {}
        for key in self.label:
            label[key] = np.concatenate(self.label[key], axis=0)
        prediction = np.transpose(self.ret, [1, 2, 0])
        target = np.transpose(label['pos_gt_src'], [1, 2, 0])

        jnt_missing = np.transpose(label['jnt_missing'], [1, 0])
        jnt = 1 - jnt_missing
        err = prediction - target
        err = np.linalg.norm(err, axis=1)

        head = np.transpose(label['headboxes_src'], [1, 2, 0])
        head = head[1, :, :] - head[0, :, :]
        head = np.linalg.norm(head, axis=0)
        head *= self.bias

        scale = np.multiply(head, np.ones((len(err), 1)))
        scaled_err = np.divide(err, scale)
        scaled_err = np.multiply(scaled_err, jnt)
        count = np.sum(jnt, axis=1)

        mask = np.multiply((scaled_err < self.threshold), jnt)
        ret = np.divide(100. * np.sum(mask, axis=1), count)
        ret = np.ma.array(ret, mask=False)
        ret.mask[6:8] = True
        return np.mean(ret) / 100

    def report(self):
        return "accuracy is %f" % (self.compute())
