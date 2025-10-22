# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
import torch
import numpy

# For OPT OP Test


@register_plugin(PluginType.Metric, '0.01')
class MaxAbsErrorMetric(OptBaseMetric):
    def __init__(self):
        self.errors = []
        self.mse = []
        self.cos = []

    def __call__(self, pred, target):
        sim_per_output = []
        mse_per_output = []
        cos_per_output = []
        for o_p, o_t in zip(pred, target):
            x = o_p.float().reshape(-1)
            y = o_t.float().reshape(-1)
            sim_per_output.append(torch.max(torch.abs(x - y)).cpu().numpy())
            mse_per_output.append(((x - y)*(x - y)).mean().cpu().numpy())
            cos_per_output.append(cosine_distance(x, y))
        self.errors.append(sim_per_output)
        self.mse.append(mse_per_output)
        self.cos.append(cos_per_output)

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
        mse = numpy.array(self.mse).T
        for i, e in enumerate(mse):
            txt += "\noutput %d: mse is %f" % (i, numpy.mean(e))
        cos = numpy.array(self.cos).T
        for i, e in enumerate(cos):
            txt += "\noutput %d: cos is %f" % (i, numpy.mean(e))
        return txt
