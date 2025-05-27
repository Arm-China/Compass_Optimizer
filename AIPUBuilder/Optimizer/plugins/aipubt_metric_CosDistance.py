# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

import torch


@register_plugin(PluginType.Metric, '1.0')
class CosDistanceMetric(OptBaseMetric):
    """
    This plugin supports the model has multi-outputs and metric cosine distance of each output
    between label and prediction.
    """

    def __init__(self):
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.sim = []

    def __call__(self, pred, target):
        batch = pred[0].shape[0]
        device = pred[0].device
        x, y = [], []
        for b in range(batch):
            batch_data = []
            for i in range(len(pred)):
                data = pred[i][b].squeeze_().reshape(-1,)
                batch_data.extend(data)
            batch_data = torch.Tensor(batch_data).to(device).double()
            x.append(batch_data)
        if isinstance(target, dict):
            for b in range(batch):
                batch_label = []
                for k in target.keys():
                    label = target[k][b].squeeze_().reshape(-1,)
                    batch_label.extend(label)
                batch_label = torch.Tensor(batch_label).to(device).double()
                y.append(batch_label)
        elif isinstance(target, list):
            for b in range(batch):
                batch_label = []
                for i in range(len(target)):
                    label = target[i][b].squeeze_().reshape(-1,)
                    batch_label.extend(label)
                batch_label = torch.Tensor(batch_label).to(device).double()
                y.append(batch_label)
        else:
            for b in range(batch):
                batch_label = target[b].to(device).squeeze_().reshape(-1).double()
                y.append(batch_label)

        for b in range(batch):
            self.sim.append(self.cos(x[b], y[b]))

    def reset(self):
        self.sim = []

    def compute(self):
        # shape of tensor self.sim is rank 1 as x, y above have iterate all dims and reshape
        t = torch.Tensor(self.sim)
        return float(torch.mean(t, 0))

    def report(self):
        return "cosine similarity is %f" % (self.compute())
