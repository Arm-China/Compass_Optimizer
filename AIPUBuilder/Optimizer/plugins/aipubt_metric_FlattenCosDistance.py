# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from AIPUBuilder.Optimizer.framework import *

import torch
import numpy

# For OPT OP Test
# It using mean value of all multi outputs of all batch


@register_plugin(PluginType.Metric, '0.01')
class FlattenCosDistanceMetric(OptBaseMetric):
    def __init__(self):
        self.cos = torch.nn.CosineSimilarity()
        self.sim = []

    def __call__(self, pred, target):
        sim_per_output = []
        for o_p, o_t in zip(pred, target):
            if len(o_p.shape):
                b = o_p.shape[0]
                x = o_p.reshape(b, -1).float()
                y = o_t.reshape(b, -1).float()
                sim = numpy.mean(self.cos(x, y).cpu().flatten().numpy())
            else:  # if output is a scalar
                x = o_p
                y = o_t[0]
                sim = (x == y)

            sim_per_output.append(sim)
        self.sim.append(sim_per_output)

    def reset(self):
        self.sim = []

    def compute(self):
        sim = numpy.array(self.sim)
        return numpy.mean(sim)

    def report(self):
        txt = ''
        sims = numpy.array(self.sim).T  # [output, per call result]
        txt += "cosine similarity is %f" % numpy.mean(sims)
        for i, sim in enumerate(sims):
            txt += "\noutput %d: cosine similarity is %f" % (i, numpy.mean(sim))
        return txt
