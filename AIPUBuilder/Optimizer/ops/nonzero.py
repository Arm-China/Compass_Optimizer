# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *

register_optype('NonZero')


@op_register(OpType.NonZero)
def nonzero(self, *args):
    self.outputs[0].betensor = torch.nonzero(self.inputs[0].betensor)
    self.outputs[0].betensor = self.outputs[0].betensor.permute(1, 0)
    self.outputs[1].betensor = torch_tensor(self.outputs[0].betensor.shape[1], self.outputs[0].device).reshape([1])
    return self.outputs[0].betensor, self.outputs[1].betensor


@quant_register(OpType.NonZero)
def nonzero_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    q_bits_activation = self.attrs["q_bits_activation"]
    max_len = max(list(inp.ir_shape))
    max_qbits = torch.log2(torch.tensor(max_len)).ceil()
    out.scale = 1.0
    out.zerop = 0
    out.qbits = max(q_bits_activation, max_qbits)
    out.dtype = bits2dtype(out.qbits, is_signed=False)
    out.qinvariant = True
    out.qmin, out.qmax = dtype2range(out.dtype)
    self.outputs[1].clone_qinfo(self.outputs[0])
