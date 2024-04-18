# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
import torch


@register_plugin(PluginType.Metric, '1.0')
class IMDBMetric(OptBaseMetric):
    """
    This IMDBMetric is used for the metric of robert-a models in Optimizer.
    accuracy = correct / total.
    half sample in total is negative sentiment, half sample is postive sentiment
    The IMDb data set  is a sentiment analysis data set (two classifications),
    test set each have 25000 samples (each sample is a movie review),
    The number of samples of the positive/the negative class (ie positive/negative) is the same, 12500/12500.
    """

    def __init__(self):
        self.correct = [0, 0]
        self.total = 0

    def __call__(self, pred, target):
        pt = torch.argmax(pred[0], 1)  #
        gt = target
        batch = pt.shape[0]
        for idx in range(batch):
            if pt == gt and gt == 0:
                self.correct[0] += 1
            elif pt == gt and gt == 1:
                self.correct[1] += 1

        self.total += batch

    def reset(self):
        self.correct = [0, 0]
        self.total = 0

    def compute(self):
        try:
            acc = float(self.correct[0]+self.correct[1]) / float(self.total)
            return acc
        except ZeroDivisionError:
            OPT_ERROR('zeroDivisionError: imdb acc total label = 0')
            return float("-inf")

    def report(self):
        return "imdb sentiment acc is %f " % (self.compute())
