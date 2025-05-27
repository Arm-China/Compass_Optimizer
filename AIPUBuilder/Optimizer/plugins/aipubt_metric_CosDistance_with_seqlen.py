# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

import torch


@register_plugin(PluginType.Metric, '1.0')
class CosDistancewseqlenMetric(OptBaseMetric):
    """
    This CosDistancewseqlenMetric is used for the metric of RNNT_encoder model in Optimizer.
    The label of metric has two elements: [label_value(tensor), actual_len(int)]. This metric will
    select the actual_len in predict_value and label_value to calculate the cosine distance.
    """

    def __init__(self):
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.sim = []

    def __call__(self, pred, target):
        preds = pred[0].cpu()
        padded_targets = target[0].cpu()
        act_len = target[1].cpu()
        targets = padded_targets
        for i in range(targets.shape[0]):
            flatten_pred = preds[i][:act_len[i]].reshape([-1])
            flatten_target = targets[i][:act_len[i]].reshape([-1])
            self.sim.append(self.cos(flatten_pred, flatten_target))

    def reset(self):
        self.sim = []

    def compute(self):
        # shape of tensor self.sim is rank 1 as x, y above have iterate all dims and reshape
        t = torch.Tensor(self.sim)
        return float(torch.mean(t, 0))

    def report(self):
        return "cosine similarity is %f" % (self.compute())
