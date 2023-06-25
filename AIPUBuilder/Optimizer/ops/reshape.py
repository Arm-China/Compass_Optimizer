# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch


@op_register(OpType.Reshape)
def reshape(self, *args):
    inp = self.inputs[0].betensor
    out = self.outputs[0]
    shape = list(out.ir_shape)

    ir_batch = self.attrs['batch_size_in_IR'] if 'batch_size_in_IR' in self.attrs else 1
    batch_axis = 0
    for idx, ss in enumerate(shape):
        if ss == ir_batch:
            shape[idx] = -1
            batch_axis = idx
            break
    # shape[0] = -1 # left batch -1
    if batch_axis != 0:
        OPT_WARN('layer_id=%s, type=%s, the batch axis is not 0, making the result unpredictable possibly' %
                 (self.attrs['layer_id'], str(self.type)), log_once=True)
    if ir_batch != 0 and len(shape) != 0:
        # have batch_dim
        shape[batch_axis] = -1
    out.betensor = torch.reshape(inp, shape)
    return out.betensor


@quant_register(OpType.Reshape)
def reshape_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    out.qmin = inp.qmin
    out.qmax = inp.qmax
