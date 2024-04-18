# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch


@register_plugin(PluginType.Metric, '1.0')
class TopKMetric(OptBaseMetric):
    """
    This TopKMetric is used for the metric of image classfication models in Optimizer.
    This plugin defaultly computes Top1.
    """

    def __init__(self, K='1'):
        self.correct = 0
        self.total = 0
        self.K = int(K)

    def __call__(self, pred, target):
        _, pt = torch.topk(pred[0].reshape([pred[0].shape[0], -1]), self.K, dim=-1)  # NHWC
        for i in range(target.numel()):
            if target[i] in pt[i]:
                self.correct += 1
        self.total += target.numel()

    def reset(self):
        self.correct = 0
        self.total = 0

    def compute(self):
        try:
            acc = float(self.correct) / float(self.total)
            return acc
        except ZeroDivisionError:
            OPT_ERROR('zeroDivisionError: Topk acc total label = 0')
            return float("-inf")

    def report(self):
        return "top-%d accuracy is %f" % (self.K, self.compute())
