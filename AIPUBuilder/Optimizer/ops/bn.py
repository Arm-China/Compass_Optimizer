# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.ops.conv import *
from AIPUBuilder.Optimizer.ops.activation import apply_with_activation
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch


@quant_register(OpType.BatchNorm)
def batch_norm_quantize(self, *args):
    linear_op_quantize(self, *args)
    absorb_input_zp_to_bias_and_compress_bias_for_aiff(self, *args)

#Cf = If * Wf + Bf
# (Cq + Zc)/Sc = ((Iq + Zq) / Si) * ((Wq + Zw)/Sw) + (Bq + Zb)/Sb
# set Sb = Si * Sw
#Cq = ((Iq+Zq) * (Wq+Zw) + (Bq + Zb)) * (Sc/Si*Sw) - Zc
# where Z is zero point, S is scale


@op_register(OpType.BatchNorm)
def batch_norm(self, *args):
    inp = self.inputs[0].betensor.clone()
    weights = self.constants["weights"].betensor.clone()
    bias = self.constants['biases'].betensor.clone()
    axis = self.get_param('axis')
    if self.quantized:
        # input's zerop has been absorbed to bias.
        # inp += self.inputs[0].zerop
        weights += self.constants["weights"].broadcast_zerop
        bias += self.constants['biases'].broadcast_zerop

    inp_dim = inp.dim()
    perm = []
    if axis != inp_dim - 1 and inp_dim > 0:
        orig_perm = [p for p in range(inp_dim)]
        perm = orig_perm[:axis] + orig_perm[axis+1:] + [orig_perm[axis]]
        inp = torch.permute(inp, perm)

    x = torch.add(torch.multiply(inp, weights), bias)
    x = apply_with_activation(self, x, *args)
    if len(perm):
        orig_perm = [p for p in range(inp_dim)]
        n_perm = orig_perm[:axis] + [orig_perm[-1]] + orig_perm[axis:-1]
        x = torch.permute(x, n_perm)
    self.outputs[0].betensor = x
    return self.outputs[0].betensor
