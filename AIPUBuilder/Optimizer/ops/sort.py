# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch


'''
# IR
layer_id=1
layer_name=sort
layer_type=Sort
layer_bottom=score
layer_bottom_shape=[1,5000]
layer_bottom_type=int16
layer_top=out_score_ptr,keep
layer_top_shape=[1,5000],[1,5000]
layer_top_type=int16,uint16
'''


@op_register(OpType.Sort)
def sort(self, *args):
    dim = self.get_param('axis', optional=True, default_value=-1)
    descending = self.get_param('descending', optional=True, default_value=True)
    inpt = self.inputs[0].betensor
    st, indices = torch.sort(inpt, dim=dim, descending=descending)
    self.outputs[0].betensor = st
    self.outputs[1].betensor = indices
    return [o.betensor for o in self.outputs]


@quant_register(OpType.Sort)
def sort_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qmin = inp.qmin
    out.qmax = inp.qmax
    out.qinvariant = inp.qinvariant
    out.qbits = inp.qbits

    q_bits_activation = self.attrs['q_bits_activation']
    out = self.outputs[1]
    out.qbits = max(16, q_bits_activation)
    out.dtype = bits2dtype(out.qbits, False or self.force_dtype_int)
    out.scale = 1.0
    out.zerop = 0.0
    out.qinvariant = True
