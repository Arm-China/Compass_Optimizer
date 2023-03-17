# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch
import numpy as np

###########################################################
# layer_id=1
# layer_name=Bitwise0
# layer_type=Bitwise
# layer_bottom=[Placeholder, placehold]
# layer_bottom_shape=[[1,89,88,8],[1,89,88,8]]
# layer_bottom_type=[int8, int8]
# layer_top=[logical]
# layer_top_shape=[[1,89,88,8]]
# layer_top_type=[int8]
# layer_top_datalayout=[NHWC]
# layer_top_scale=[1.000000]
# layer_top_zp=[0]
# method=NOT

# input1 [-2 -1 0 6]
# input2 [ 3 -5 9 7]
# bitwise_and: [ 2 -5 0 6]
# bitwise_or: [-1 -1 9 7]
# bitwise_xor: [-3 4 9 1]
###########################################################
register_optype('Bitwise')


@quant_register(OpType.Bitwise)
def bitwise_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp_qinvariant = []
    for i, inp in enumerate(self.inputs):
        inp_qinvariant.append(inp.qinvariant)
    out = self.outputs[0]

    str_type = self.attrs['layer_top_type_original'][0]
    out_dtype = str2dtype(str_type)
    if False in inp_qinvariant:
        OPT_FATAL("Only support unquantized input, but inputs of op{%s} have quantized input." % (self.type))
    else:  # inp0.qinvariant is true and inp1.qinvariant is true
        qbits = dtype2bits(out_dtype)
        out.qbits = qbits
        out.scale = 1
        out.zerop = 0
        qmin, qmax = dtype2range(out_dtype)
        out.qmin = qmin
        out.qmax = qmax
        out.dtype = out_dtype
        out.qinvariant = True


@op_register(OpType.Bitwise)
def bitwise(self, *args):
    func_args = []
    inp0 = self.inputs[0].betensor.long()
    inp0_np_dtype = dtype2nptype(self.inputs[0].dtype)
    input0 = inp0.cpu().numpy().astype(inp0_np_dtype)
    func_args.append(input0)
    if len(self.inputs) > 1:
        inp1 = self.inputs[1].betensor.long()
        inp1_np_dtype = dtype2nptype(self.inputs[1].dtype)
        input1 = inp1.cpu().numpy().astype(inp1_np_dtype)
        func_args.append(input1)

    out = self.outputs[0]
    out_dtype = out.dtype
    np_out_dtype = dtype2nptype(out_dtype)
    method = self.get_param('method').upper()

    method_d = {
        "XOR": np.bitwise_xor,
        "AND": np.bitwise_and,
        "OR":  np.bitwise_or,
        "NOT": np.bitwise_not,
    }

    if method not in method_d:
        OPT_FATAL("unsupported method: %s for Bitwise in node:%s" % (method, self.name))
    else:
        outp = method_d[method](*func_args)

    outp = outp.astype(np_out_dtype)
    pytensor = PyTensor('out', outp)
    outp = pytensor.betensor.to(inp0.device)
    self.outputs[0].betensor = outp

    return self.outputs[0].betensor
