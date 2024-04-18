# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

import torch
import numpy


@register_plugin(PluginType.Metric, '1.0')
class MaxAbsErrorwseqlenMetric(OptBaseMetric):
    """
    This MaxAbsErrorwseqlenMetric is used for the metric of RNNT_encoder model in Optimizer.
    The label of metric has two elements: [label_value(tensor), actual_len(int)]. This metric will
    select the actual_len in predict_value and label_value to calculate the maximum absolute error.
    """

    def __init__(self):
        self.errors = []

    def __call__(self, pred, target):
        sim_per_output = []
        for o_p, o_t, o_len in zip(pred, target[0], target[1]):
            o_len = o_len.cpu()
            x = o_p.float()[:, :o_len].reshape(-1)
            y = o_t.float()[:o_len].reshape(-1)
            sim_per_output.append(torch.max(torch.abs(x - y)).cpu().numpy())
        self.errors.append(sim_per_output)

    def reset(self):
        self.errors = []

    def compute(self):
        errors = numpy.array(self.errors)
        return numpy.mean(errors)

    def report(self):
        txt = ''
        errors = numpy.array(self.errors).T  # [output, per call result]
        txt += "maximum absolute error is %f" % numpy.mean(errors)
        for i, e in enumerate(errors):
            txt += "\noutput %d: maximum absolute error is %f" % (i, numpy.mean(e))
        return txt
