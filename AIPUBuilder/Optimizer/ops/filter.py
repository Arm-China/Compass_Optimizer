# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.logger import *
import torch

'''
Filter Op IR:
layer_id=3
layer_name=filter_1
layer_type=Filter
layer_bottom=[Placeholder,Placeholder_1,selector]
layer_bottom_shape=[[50,6,6,6],[50,6,6,6],[50]]
layer_bottom_type=[float32,float32,float32]
layer_top=[filter_1_0,filter_1_1,effective_len]
layer_top_shape=[[50,6,6,6],[50,6,6,6],[]]
layer_top_type=[float32,float32,int32]
num=2
axis=0
'''


@op_register(OpType.Filter)
def filter(self, *args):
    """
    using the non-zero value in selector to filter the other inputs, and then get the corresponding outputs.
    """
    num = self.get_param('num')
    assert num == len(self.inputs)-1, OPT_ERROR('please check the num parameters, which %d != len(inputs)-1' % (num))
    axis = self.get_param('axis')

    selector = self.inputs[-1].betensor
    nonzero_idx = torch.nonzero(selector, as_tuple=True)[0]
    for i, bt in enumerate(self.inputs[:-1]):
        init_t = torch.zeros_like(bt.betensor)
        selected_inpt = torch.index_select(bt.betensor, dim=axis, index=nonzero_idx)
        src_unbind = torch.unbind(selected_inpt, dim=axis)
        dst_unbind = torch.unbind(init_t, dim=axis)
        for s in range(len(src_unbind)):
            dst_unbind[s][...] = src_unbind[s][...]
        self.outputs[i].betensor = torch.cat([torch.unsqueeze(dt, axis) for dt in dst_unbind], dim=axis)

    self.outputs[-1].betensor = torch.tensor(len(nonzero_idx))

    return [o.betensor for o in self.outputs]


@quant_register(OpType.Filter)
def quantize_filter(self, *args):
    for inp, out in zip(self.inputs[:-1], self.outputs):
        out.dtype = inp.dtype
        out.scale = inp.scale
        out.zerop = inp.zerop
        out.qbits = inp.qbits
        out.qinvariant = inp.qinvariant

    q_bits_activation = self.attrs['q_bits_activation']
    out = self.outputs[-1]
    out.qbits = max(16, q_bits_activation)
    out.dtype = bits2dtype(out.qbits, False or self.force_dtype_int)
    out.scale = 1.0
    out.zerop = 0.0
    out.qinvariant = True
