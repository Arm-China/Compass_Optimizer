# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils.quant_tool_utils import *
from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.framework import *

import torch

'''
*reference: https://www.cnblogs.com/liushengchieh/p/14971752.html
*IR:
layer_id=11
layer_name=ReverseV2
layer_type=ReverseSequence
layer_bottom=[bidirectional_rnn/bw/bw/transpose_1_0,ReverseV2_seq_len]
layer_bottom_shape=[[1,97,800],[1]]
layer_bottom_type=[float32,int32]
layer_top=[ReverseV2_0]
layer_top_shape=[[1,97,800]]
layer_top_type=[float32]
batch_axis=0
time_axis=1

'''


def local_reserve_sequence(inp, lengths, batch_axis, time_axis):

    out = torch.zeros_like(inp, device=inp.device)
    for b in range(lengths.shape[0]):
        seq_len = lengths[b].int()
        temp = torch.index_select(inp, batch_axis, torch.tensor(b, device=inp.device))
        flip_part = torch.index_select(temp, time_axis, torch.arange(seq_len, device=inp.device)).flip(dims=[time_axis])
        reserve_part = torch.index_select(temp, time_axis, torch.arange(
            temp.shape[time_axis] - seq_len, device=inp.device)+seq_len)
        one_out = torch.cat([flip_part, reserve_part], dim=time_axis)
        out = torch.cat([out, one_out], dim=batch_axis)
    out = torch.split(out, inp.shape[batch_axis], dim=batch_axis)[-1]
    return out


@op_register(OpType.ReverseSequence)
def reverse_sequence(self, *args):
    inpt = self.inputs[0].betensor

    batch_axis = self.get_param('batch_axis')
    time_axis = self.get_param('time_axis')
    if batch_axis == time_axis:
        (OPT_WARN('layer_id=%s, type=%s, please check batch_axis and time_axis, which now batch_axis=%d, time_axis=%d' %
                  (self.attrs['layer_id'], str(self.type), batch_axis, time_axis)))

    # The length of sequence_lens = input_shape[batch_axis]
    if len(self.inputs) == 1:
        seq_lent = torch.tensor([inpt.shape[time_axis]] * inpt.shape[batch_axis], device=inpt.device)
    else:
        seq_shape = self.inputs[1].betensor.shape[0]
        if seq_shape != inpt.shape[batch_axis]:
            OPT_WARN('The length of sequence_lens must equal to input_shape[batch_axis]')
            seq_lent = torch.full([inpt.shape[batch_axis]], self.inputs[1].betensor[0].int().item())
        else:
            seq_lent = self.inputs[1].betensor
        # The value of sequence_lens <= input_shape[time_axis]
        if torch.max(self.inputs[1].betensor) > inpt.shape[time_axis]:
            OPT_WARN('The value of sequence_lens must not greater than input_shape[time_axis]')
            limited_range = torch.tensor(inpt.shape[time_axis], dtype=torch.float32,
                                         device=self.inputs[1].betensor.device)
            self.inputs[1].betensor = torch.where(
                self.inputs[1].betensor > inpt.shape[time_axis], limited_range, self.inputs[1].betensor)

    outt = local_reserve_sequence(inpt, seq_lent, batch_axis, time_axis)
    self.outputs[0].betensor = outt
    return outt


@quant_register(OpType.ReverseSequence)
def reverse_sequence_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qmin = inp.qmin
    out.qmax = inp.qmax
    out.qinvariant = inp.qinvariant
    out.qbits = inp.qbits
