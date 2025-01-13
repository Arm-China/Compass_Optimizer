# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *

register_optype('IsNaN')


@op_register(OpType.IsNaN)
def isnan(self, *args):
    out = torch.isnan(self.inputs[0].betensor).int()
    self.outputs[0].betensor = PyTensor('out', out, self.outputs[0].dtype).betensor
    return self.outputs[0].betensor


@quant_register(OpType.IsNaN)
def isnan_quantize(self, *args):
    out = self.outputs[0]
    out.scale = 1.0
    out.zerop = 0
    out.qbits = 8
    out.dtype = bits2dtype(out.qbits, is_signed=False)
    out.qinvariant = True
    out.qmin, out.qmax = dtype2range(out.dtype)
