# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *

register_optype('IsInf')


@op_register(OpType.IsInf)
def isinf(self, *args):
    detect_negative = self.get_param('detect_negative', optional=True, default_value=True)
    detect_positive = self.get_param('detect_positive', optional=True, default_value=True)

    inp = self.inputs[0].betensor
    mask = torch.zeros_like(inp, device=inp.device).bool()
    if detect_positive:
        positive_mask = (inp == float('inf'))
        mask = torch.bitwise_or(mask, positive_mask)
    if detect_negative:
        negative_mask = (inp == float('-inf'))
        mask = torch.bitwise_or(mask, negative_mask)
    self.outputs[0].betensor = PyTensor('out', mask.int(), self.outputs[0].dtype).betensor
    return self.outputs[0].betensor


@quant_register(OpType.IsInf)
def isinf_quantize(self, *args):
    out = self.outputs[0]
    out.scale = 1.0
    out.zerop = 0
    out.qbits = 8
    out.dtype = bits2dtype(out.qbits, is_signed=False)
    out.qinvariant = True
    out.qmin, out.qmax = dtype2range(out.dtype)
