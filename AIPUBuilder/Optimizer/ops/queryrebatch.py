# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *

register_optype('QueryRebatch')


@op_register(OpType.QueryRebatch)
def queryrebatch_forward(self, *args):
    inputs = self.inputs
    if len(inputs) - 1 != self.outputs[0].ir_shape[1]:
        OPT_ERROR(f"please check the queryrebatch IR, the len(input) - 1 should be equal to output[0].shape[1]")

    max_len = max([t.betensor.shape[0] for t in self.inputs[1:]])
    output_shape = list(self.outputs[0].ir_shape)
    output_shape[2] = max_len
    self.outputs[0].betensor = torch.zeros(output_shape).to(self.inputs[0].device)
    for i, inp_t in enumerate(self.inputs[1:]):
        self.outputs[0].betensor[:, i, :inp_t.betensor.shape[0]
                                 ] = self.inputs[0].betensor[:, self.inputs[i+1].betensor.long()]

    return self.outputs[0].betensor


@quant_register(OpType.QueryRebatch)
def queryrebatch_quantize(self, *args):
    self.outputs[0].clone_qinfo(self.inputs[0])
