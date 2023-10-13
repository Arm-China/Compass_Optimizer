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
        w_zp = self.constants["weights"].zerop
        w_zshape = [1] * weights.dim()
        w_zshape[0] = -1
        weights += w_zp.reshape(w_zshape) if isinstance(w_zp, torch.Tensor) else w_zp
        bias += self.constants['biases'].zerop

    inp_dim = inp.dim()
    perm = []
    if axis != inp_dim - 1 and inp_dim > 0:
        orig_perm = [p for p in range(inp_dim)]
        perm = orig_perm[:axis] + orig_perm[axis+1:] + [orig_perm[axis]]
        inp = torch.permute(inp, perm)

    x = torch.add(torch.multiply(inp, weights), bias)

    requant_scale = 1
    requant_shift = 0
    if self.quantized:
        if 'scale_value' in self.params:
            requant_scale = self.params['scale_value']
        elif "scale" in self.constants:
            requant_scale = self.constants["scale"].betensor

        if 'shift_value' in self.params:
            requant_shift = self.params['shift_value']
        elif "shift" in self.constants:
            requant_shift = self.constants["shift"].betensor

    x = apply_with_activation(self, x,
                              self.inputs[0].scale * self.constants["weights"].scale, 0,
                              requant_scale,
                              requant_shift,
                              *args)
    if len(perm):
        orig_perm = [p for p in range(inp_dim)]
        n_perm = orig_perm[:axis] + [orig_perm[-1]] + orig_perm[axis:-1]
        x = torch.permute(x, n_perm)

    self.outputs[0].betensor = x
    return x
