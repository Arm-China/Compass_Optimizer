# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *


@op_register(OpType.ArgMinMax)
def argminmax(self, *args):
    axis_ = self.get_param('axis')
    method_ = self.get_param("method").upper()
    select_last_index_ = self.get_param("select_last_index")
    if method_ not in ['MAX', 'MIN']:
        OPT_FATAL("please check method in argminmax op, which supports [MAX, MIN]")
    inp = self.inputs[0].betensor
    out = self.outputs[0]
    if select_last_index_:
        inp = torch.flip(inp, dims=[axis_])

    if method_ == 'MAX':
        out.betensor = torch.argmax(inp, dim=axis_, keepdim=True)
    elif method_ == 'MIN':
        out.betensor = torch.argmin(inp, dim=axis_, keepdim=True)

    if select_last_index_:
        out.betensor = inp.shape[axis_] - out.betensor - 1
    return out.betensor


@quant_register(OpType.ArgMinMax)
def argminmax_quantize(self, *args):
    q_bits_activation = self.attrs["q_bits_activation"]
    out = self.outputs[0]
    out.scale = 1.
    out.zerop = 0
    out.qbits = max(16, q_bits_activation)
    out.dtype = bits2dtype(out.qbits, is_signed=False or self.force_dtype_int)
    out.qinvariant = True
