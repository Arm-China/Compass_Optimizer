# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch


@op_register(OpType.ZeroFraction)
def zerofraction(self, *args):
    inp_t = self.inputs[0].betensor + self.inputs[0].zerop
    total_num = inp_t.numel()
    nonzero_num = torch.count_nonzero(inp_t)
    zero_num = total_num - nonzero_num
    if self.quantized:
        enlarge_scale = self.get_param('output_scale')
        zero_fraction = (zero_num * enlarge_scale + total_num//2) // total_num
    else:
        zero_fraction = zero_num / total_num
    self.outputs[0].betensor = torch.tensor(zero_fraction - self.outputs[0].zerop)
    return self.outputs[0].betensor


@quant_register(OpType.ZeroFraction)
def zerofraction_quantize(self, *args):
    q_bits_activation = self.attrs["q_bits_activation"]
    enlarge_scale = 2 ** q_bits_activation - 1
    self.params['output_scale'] = enlarge_scale

    out = self.outputs[0]
    out.qinvariant = False
    out.qbits = q_bits_activation
    out.dtype = bits2dtype(q_bits_activation, False or self.force_dtype_int)
    out.qmin, out.qmax = dtype2range(out.dtype)
    out.scale = enlarge_scale
    out.zerop = 0
    if self.force_dtype_int:
        out.zerop = - out.qmin
