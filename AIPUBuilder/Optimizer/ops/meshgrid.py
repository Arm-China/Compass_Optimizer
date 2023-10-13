# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch

###########################################################
# layer_id=1
# layer_name=Meshgrid
# layer_type=Meshgrid
# layer_bottom=x,y
# layer_bottom_shape=[[3],[4]]
# layer_bottom_type=int8,int8
# layer_top=Sigmoid_0
# layer_top_shape=[4,3][4,3]
# layer_top_type=int8,int8
# indexing=xy/ij
# sparse= false/true
# copy=false/true
###########################################################


@quant_register(OpType.Meshgrid)
def meshgrid_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp0 = self.inputs[0]
    inp1 = self.inputs[1]

    # currently only support int input
    if inp0.qinvariant and inp1.qinvariant:
        for idx, out in enumerate(self.outputs):
            inp = self.inputs[idx]
            out.scale = inp.scale
            out.zerop = inp.zerop
            out.qbits = inp.qbits
            out.qmin, out.qmax = inp.qmin, inp.qmax
            out.dtype = inp.dtype
            out.qinvariant = True
    else:
        OPT_FATAL('layer_id=%s, type=%s, currently only support int input and qinvariant is true' % (
            self.attrs['layer_id'], str(self.type)))


@op_register(OpType.Meshgrid)
def meshgrid(self, *args):
    if len(self.inputs) > 2:
        OPT_FATAL('layer_id=%s, type=%s, currently only support two inputs, now input num is %s' % (
            self.attrs['layer_id'], str(self.type), str(len(self.inputs))))

    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    indexing = self.get_param('indexing')
    sparse = self.get_param('sparse', optional=True, default_value=False)
    copy = self.get_param('copy', optional=True, default_value=True)

    if indexing not in ['xy', 'ij']:
        OPT_WARN('layer_id=%s, type=%s, indexing only support "xy" or "ij", now indexing is %s, use "xy" instead' % (
            self.attrs['layer_id'], str(self.type), indexing))
        indexing = 'xy'

    input_x = inp0.betensor
    input_y = inp1.betensor

    output0, output1 = torch.meshgrid(input_x, input_y, indexing=indexing)

    if sparse:
        if indexing == 'xy':
            output0 = output0[:1, ...]
            output1 = output1[..., :1]
        else:  # 'ij'
            output0 = output0[..., :1]
            output1 = output1[:1, ...]

    self.outputs[0].betensor = output0
    self.outputs[1].betensor = output1

    return output0, output1
