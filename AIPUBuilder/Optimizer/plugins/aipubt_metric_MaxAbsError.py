# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

import torch
import numpy

# For OPT OP Test


@register_plugin(PluginType.Metric, '0.01')
class MaxAbsErrorMetric(OptBaseMetric):
    def __init__(self):
        self.errors = []

    def __call__(self, pred, target):
        sim_per_output = []
        for o_p, o_t in zip(pred, target):
            x = o_p.float().reshape(-1)
            y = o_t.float().reshape(-1)
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
