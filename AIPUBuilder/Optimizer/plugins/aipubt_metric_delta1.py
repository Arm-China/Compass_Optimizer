# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch
import math


@register_plugin(PluginType.Metric, '1.0')
class delta1Metric(OptBaseMetric):
    """
    This delta1Metric is used for the metric of fast_depth_onnx model in Optimizer.
    """

    def __init__(self):
        self.num = 0
        self.delta1_sum = 0

    def __call__(self, pred, target):
        prediction = pred[0]
        batch_size = pred[0].shape[0]
        mask = ((target > 0) + (prediction > 0)) > 0

        prediction = prediction[mask] * 1000
        target = target[mask] * 1000

        max_ratio = torch.max(prediction / target, target / prediction)
        delta1 = float((max_ratio < 1.25).float().mean())

        self.num += batch_size
        self.delta1_sum += batch_size * delta1

    def reset(self):
        self.num = 0
        self.delta1_sum = 0

    def compute(self):
        ret = self.delta1_sum / self.num
        return ret

    def report(self):
        return "delta1 accuracy is %f" % (self.compute())
