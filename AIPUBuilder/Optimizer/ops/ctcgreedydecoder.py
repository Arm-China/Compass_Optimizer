# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.logger import OPT_WARN
import torch

'''
layer_id=143
layer_name=CTCGreedyDecoder
layer_type=CTCGreedyDecoder
layer_bottom=[CTCGreedyDecoder_Transpose,seq_len_0]
layer_bottom_shape=[[1,390,4335],[1]]
layer_bottom_type=[float32,int32]
layer_top=[CTCGreedyDecoder_0]
layer_top_shape=[[1,4096,1,1]]
layer_top_type=[int32]
'''


def remove_blank(indexes, blanks=[0]):
    # remove blank
    no_blank_index = indexes
    for blank in blanks:
        no_blank_index = no_blank_index[torch.where(no_blank_index != blank)]
    return no_blank_index


def merge_repeated(indexes):
    '''
        index = [1,2,3,3,3,4,4,7,8,8]
        index_pad = [1,2,3,3,3,4,4,7,8,8,-1]
        pad_index = [-1,1,2,3,3,3,4,4,7,8,8]
        diff = index_pad - pad_index
        diff = [2,1,1,0,0,1,0,1,0,-9]
        diff = diff[:-1]
        according to the nonzero element idx of diff, gather from index, then get merged_repeated result
    '''
    dev = indexes.device
    raw_shape = indexes.shape
    pad_shape = [*raw_shape[:-1], 1]
    pad_index = torch.full(pad_shape, -1, device=dev, dtype=torch.int32)

    indexes_pad = torch.cat([indexes, pad_index], dim=-1)
    pad_indexes = torch.cat([pad_index, indexes], dim=-1)

    diff = indexes_pad - pad_indexes
    diff_no_pad = diff[..., :-1]
    merge_index = indexes[torch.where(diff_no_pad != 0)]
    return merge_index


@op_register(OpType.CTCGreedyDecoder)
def ctcgreedydecoder(self, *args):
    """
    :param self:
    :param args: input0: [batch_size, num_time_steps, num_classes], input1: [act_seq_len]
    :return:
    """
    inps = self.inputs[0].betensor
    seq_lens = self.inputs[1].betensor
    is_merge_repeated = self.get_param('merge_repeated')
    batch = inps.shape[0]
    tout = torch.full([batch] + [*self.outputs[0].ir_shape][1:], dtype2range(self.outputs[0].dtype)[1])
    for b in range(batch):
        inp = inps[b]
        seq_len = seq_lens[b].int().item()
        if seq_len <= 0:
            OPT_WARN('id=%s, type=%s, please check the input1: seq_len=%d' % (
                self.attrs['layer_id'], str(self.type), seq_len))
            seq_len = inp.shape[0]
        act_inp = inp[:seq_len, :]
        num_classes = inp.shape[1]
        max_idx = torch.argmax(act_inp, dim=-1)
        if is_merge_repeated:
            out = merge_repeated(max_idx)
        else:
            out = max_idx
        out = remove_blank(out, blanks=[num_classes - 1])
        out = out.reshape([*out.shape, 1, 1])
        tout[b, :out.shape[0], :, :] = out
    self.outputs[0].betensor = tout
    return tout


@quant_register(OpType.CTCGreedyDecoder)
def ctcgreedydecoder_quantize(self, *args):
    q_bits_activation = self.attrs['q_bits_activation']
    out = self.outputs[0]
    out.qbits = max(16, q_bits_activation)
    out.dtype = bits2dtype(out.qbits, False or self.force_dtype_int)
    out.scale = 1.0
    out.zerop = 0.0
    out.qinvariant = True
