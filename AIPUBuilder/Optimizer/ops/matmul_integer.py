# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch

###########################################################
# layer_id=3
# layer_name=MatMulInteger_0
# layer_type=MatMulInteger
# layer_bottom=[Placeholder_0_post_reshape,Placeholder_1]
# layer_bottom_shape=[[1,1,3],[2,3,4]]
# layer_bottom_type=[int8,int8]
# layer_top=[MatMulInteger_0]
# layer_top_shape=[[2,1,4]]
# layer_top_type=[int32]
# a_zero_point=82
# b_zero_point=-25
###########################################################


@quant_register(OpType.MatMulInteger)
def matmulinteger_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]

    # currently only support int input
    if inp0.qinvariant and inp1.qinvariant:
        out.scale = 1
        out.zerop = 0
        out.qbits = 32
        out.qmin, out.qmax = bits2range(out.qbits, True)
        out.dtype = bits2dtype(out.qbits, True)
        out.qinvariant = True
    else:
        OPT_FATAL('layer_id=%s, type=%s, currently only support int input and qinvariant is true' % (
            self.attrs['layer_id'], str(self.type)))


@op_register(OpType.MatMulInteger)
def matmulinteger(self, *args):
    if len(self.inputs) > 2:
        OPT_FATAL('layer_id=%s, type=%s, currently only support two inputs, now input num is %s' % (
            self.attrs['layer_id'], str(self.type), str(len(self.inputs))))

    inp0 = self.inputs[0].betensor
    inp1 = self.inputs[1].betensor
    a_zero_point = self.get_param('a_zero_point')
    b_zero_point = self.get_param('b_zero_point')

    X0 = inp0 - a_zero_point
    X1 = inp1 - b_zero_point

    output = torch.matmul(X0, X1)

    self.outputs[0].betensor = output

    return output
