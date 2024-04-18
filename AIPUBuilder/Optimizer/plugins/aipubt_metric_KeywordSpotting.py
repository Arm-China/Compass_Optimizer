# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch


@register_plugin(PluginType.Metric, '1.0')
class KeywordSpottingMetric(OptBaseMetric):
    """
    This KeywordSpottingMetric is used for the metric of kws_gru/kws_lstm models in Optimizer.
    accuracy = correct / total.
    """

    def __init__(self, K=1):
        self.correct = 0
        self.total = 0

    def __call__(self, pred, target):
        _, pt = torch.topk(pred[0], 1, dim=-1)  # NHWC
        _, gt = torch.topk(target, 1, dim=-1)  # NHWC
        batch = pt.shape[0]
        for idx in range(batch):
            if pt[idx][0] == gt[idx][0]:
                self.correct += 1
        self.total += batch

    def reset(self):
        self.correct = 0
        self.total = 0

    def compute(self):
        try:
            acc = float(self.correct) / float(self.total)
            return acc
        except ZeroDivisionError:
            OPT_ERROR('zeroDivisionError: kws acc total label = 0')
            return float("-inf")

    def report(self):
        return "accuracy is %f" % (self.compute())
