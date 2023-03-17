# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
register_optype('FakeQuantWithMinMaxVars')


@op_register(OpType.FakeQuantWithMinMaxVars)
# this op only float32 can change output trivially,from op definition, it should not have real quantization
def fake_quant_with_minmax_vars(self, *args):
    inp = self.inputs[0].betensor
    if self.quantized:
        self.outputs[0].betensor = inp
    else:
        min = torch.tensor(self.get_param('min'))
        max = torch.tensor(self.get_param('max'))
        narrow_range = self.get_param('narrow_range')
        num_bits = self.get_param('num_bits')
        # Before quantization, min and max values are adjusted with the following logic
        if min > 0:
            min_adj = 0
            max_adj = max-min
        elif max < 0:
            min_adj = min-max
            max_adj = 0
        else:
            scale = (max-min)/((1 << num_bits)-1)
            min_adj = scale*torch.round(min/scale)
            max_adj = max + min_adj - min
        # inputs values are quantized into the quantization range ( [0; 2^num_bits - 1]
        # when narrow_range is false and [1; 2^num_bits - 1] when it is true)
        # and then de-quantized and output as floats in [min; max] interva
        scale = ((1 << num_bits)-1)/(max_adj-min_adj)
        if narrow_range:
            qmin = 1
        else:
            qmin = 0
        qmax = (1 << num_bits)-1
        quan = linear_quantize_clip(inp, scale, 0, qmin, qmax)
        out = linear_dequantize(quan, scale, 0)
        out = torch.clamp(out, min_adj, max_adj)

        self.outputs[0].betensor = out
    return self.outputs[0].betensor


@quant_register(OpType.FakeQuantWithMinMaxVars)
def fake_quant_with_minmax_vars_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    # min = self.get_param('min')
    # max = self.get_param('max')
    # narrow_range = self.get_param('narrow_range')
    # num_bits = self.get_param('num_bits')
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.qinvariant = inp.qinvariant
