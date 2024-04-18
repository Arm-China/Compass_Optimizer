# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
import torch


@register_plugin(PluginType.Metric, '2.0')
class MyTopKMetric(OptBaseMetric):
    def __init__(self, K='1'):
        self.correct = 0
        self.total = 0
        self.K = int(K)
        OPT_INFO(f'Customized metric plugin is enabled. k={K}')

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
