# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch


##############floatIR#########################
# layer_id=2
# layer_name=SquaredDifference1
# layer_type=Compress
# layer_bottom=[fingerprint_input_0, index]
# layer_bottom_shape=[[1,2,2,3], [4]]
# layer_bottom_type=[float32, int32]
# layer_top=[output]
# layer_top_shape=[[1,2,1,3]]
# layer_top_type=[float32]
# axis=2

@quant_register(OpType.Compress)
def compress_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    out = self.outputs[0]

    out.scale = inp.scale
    out.zerop = inp.zerop
    out.dtype = inp.dtype
    out.qbits = inp.qbits
    out.qmin = inp.qmin
    out.qmax = inp.qmax
    out.qinvariant = inp.qinvariant


@op_register(OpType.Compress)
def compress(self, *args):
    inp0 = self.inputs[0].betensor
    inp1 = self.inputs[1].betensor
    out = self.outputs[0]

    if inp1.dim() != 1:
        OPT_FATAL('layer_id=%s, type=%s, condition rank does not equal 1-D, please check' % (
            self.attrs['layer_id'], str(self.type)))

    axis = int(self.get_param('axis'))  # only int

    index = inp1.long()
    input_shape = list(inp0.shape)
    axis_dim = input_shape[axis]
    if index.numel() > axis_dim:
        OPT_WARN('layer_id=%s, type=%s, len(condition) is larger than the size of a along the given axis, so data slices exceeding the condition length are discarded' % (
            self.attrs['layer_id'], str(self.type)))

    input_shape[axis] = 0
    data = torch.zeros(input_shape, device=inp0.device)
    for idx, indics in enumerate(index):
        if idx >= axis_dim:
            break
        if indics:
            b = torch.index_select(
                inp0, dim=axis, index=torch.tensor(idx, device=inp1.device))
            data = torch.cat((data, b), axis)
    out.betensor = data

    return out.betensor
