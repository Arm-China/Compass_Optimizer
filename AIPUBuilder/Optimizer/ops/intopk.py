# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *

register_optype('InTopK')


@op_register(OpType.InTopK)
def intopk(self, *args):
    k = self.get_param('k')

    largest = self.get_param("largest", optional=True, default_value=True)
    issorted = self.get_param("sorted", optional=True, default_value=True)

    inp_betensors1 = self.inputs[0].betensor
    inp_betensors2 = self.inputs[1].betensor
    if self.inputs[1].betensor.ndim > 1:
        inp_betensors2 = torch.squeeze(self.inputs[1].betensor)  # rank 1
    k = min(k, inp_betensors1.shape[-1])
    topk_v, topk_indice = torch.topk(inp_betensors1, k, -1, largest, issorted)
    if((inp_betensors1.shape[0]) != len(inp_betensors2)):
        OPT_FATAL("target must have the same size as input along predict'axis  ")
    out = []
    for i in range(len(inp_betensors2)):
        out.append(inp_betensors1[i, int(inp_betensors2[i])] in topk_v[i, :])
    self.outputs[0].betensor = torch.tensor(out)
    return self.outputs[0].betensor


@quant_register(OpType.InTopK)
def intopk_quantize(self, *args):
    out = self.outputs[0]
    out.scale = 1.0
    out.zerop = 0
    out.dtype = Dtype.INT8 if self.force_dtype_int else Dtype.UINT8
    out.qbits = dtype2bits(out.dtype)
    out.qinvariant = True
