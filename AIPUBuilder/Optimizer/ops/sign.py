# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *


@op_register(OpType.Sign)
def sign(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.betensor = torch.sign((inp.betensor.float() + inp.zerop) if self.quantized else inp.betensor)  # -1, 0, 1
    return out.betensor


@quant_register(OpType.Sign)
def sign_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.scale = 1
    out.zerop = 0
    out.qbits = inp.qbits
    out.dtype = bits2dtype(out.qbits, is_signed=True, use_float=False)
    out.qinvariant = True
