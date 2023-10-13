# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *


@op_register(OpType.Split)
# IR
# layer_type=Split
# layer_bottom=rpn_class/concat_0
# layer_bottom_shape=[1,261888,2]
# layer_bottom_type=float32
# layer_top=split_out0,split_out1
# layer_top_shape=[1,261888,1],[1,261888,1]
# layer_top_type=float32,float32
# axis=2
# num_split=2
def split(self, *args):
    axis = self.get_param('axis')
    inp_betensors = self.inputs[0].betensor
    split_sizes = self.get_param('splits')
    out = torch.split(inp_betensors, split_sizes, dim=axis)

    for i, outp in enumerate(out):
        self.outputs[i].betensor = outp
    return out


@quant_register(OpType.Split)
def split_quantize(self, *args):
    for i, out in enumerate(self.outputs):
        inp = self.inputs[0]
        out = self.outputs[i]
        out.dtype = inp.dtype
        out.scale = inp.scale
        out.zerop = inp.zerop
        out.qbits = inp.qbits
        out.qinvariant = inp.qinvariant
