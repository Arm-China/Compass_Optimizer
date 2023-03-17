# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.plugins.aipubt_metric_mIoU import mIoUMetricBase
import torch
import cv2
import numpy as np


@register_plugin(PluginType.Metric, '1.0')
class F1scoreMetric(mIoUMetricBase):
    """
    This F1scoreMetric is used for the metric of onnx_sne_roadseg models in Optimizer.
    This plugin computes the f1-measure metric for kitti dataset.
    """

    def __init__(self, layout='NHWC'):
        super().__init__()
        if layout == 'NCHW':
            self.channel_axis = 1

    def __call__(self, pred, target):
        if isinstance(target, list):
            target = target[0]
        super().__call__(pred, target)

    def reset(self):
        super().reset()

    def compute(self):
        conf = self.confusion_matrix
        pred = (np.diag(conf) / conf.sum(0).astype(np.float))[1]
        recall = (np.diag(conf) / conf.sum(1).astype(np.float))[1]
        f1score = 2*(recall*pred)/(recall+pred)
        return f1score

    def report(self):
        return "F1 score is %f" % (self.compute())
