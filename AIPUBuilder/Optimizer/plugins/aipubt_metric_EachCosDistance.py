# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch
import numpy as np


@register_plugin(PluginType.Metric, '1.0')
class EachCosDistanceMetric(OptBaseMetric):
    """
    This EachCosDistanceMetric is used for the metric of RNNT_decoder model in Optimizer.
    This plugin supports the model has multi-outputs and metric cosine distance of each output
    between label and prediction.
    """

    def __init__(self):
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.sim = None
        self.total_quan = 0
        self.total_sim = None

    def __call__(self, pred, target):
        if self.sim is None:
            self.sim = torch.zeros([len(pred)])
        pred_list = []
        target_list = []
        device = pred[0].device
        output_num = len(pred)

        for i in range(output_num):
            pred_list.append(pred[i].reshape(-1).double().to(device))

        if isinstance(target, dict):
            for k in target.keys():
                target_list.append(target[k].reshape(-1,).double().to(device))
        elif isinstance(target, list):
            for i in range(output_num):
                target_list.append(target[i].reshape(-1,).double().to(device))
        else:
            target_list.append(target.reshape(-1,).double().to(device))

        for i in range(output_num):
            x = pred_list[i]
            y = target_list[i]
            if torch.count_nonzero(x) == 0 and torch.count_nonzero(y) == 0:
                self.sim[i] += 1.0
            else:
                self.sim[i] += self.cos(x, y).cpu().numpy()

        self.total_quan += 1

    def reset(self):
        self.sim = None

    def compute(self):
        self.total_sim = self.sim.cpu().numpy()/self.total_quan
        return float(np.mean(self.total_sim, 0))

    def report(self):
        total_cosine = self.compute()
        total_str = 'cosine similarity is: \n'
        for i in range(len(self.total_sim)):
            total_str += '\toutput_%s cosine similarity : %s\n' % (
                i, self.total_sim[i])

        return "cosine similarity is %f.\n%s" % (total_cosine, total_str)
