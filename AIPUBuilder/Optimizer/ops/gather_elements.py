# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import *


# IR
# layer_id=2
# layer_name=gather
# layer_type=GatherElements
# layer_bottom=[params,ids]
# layer_bottom_shape=[[1,16320],[1,5]]
# layer_bottom_type=[float,int32]
# layer_top=[gather]
# layer_top_shape=[[1,5]]
# layer_top_type=[float]
# axis=1


# onnx gather_elements is same as torch gather
@op_register(OpType.GatherElements)
def gather_elements(self, *args):
    indice_betensor = self.inputs[1].betensor.clone()
    inp0_betensors = self.inputs[0].betensor

    if inp0_betensors.dim() != indice_betensor.dim():
        OPT_FATAL('layer_id=%s, type=%s, inp0 and inp1 rank are not same, please check' % (
            self.attrs['layer_id'], str(self.type)))

    axis = self.get_param('axis')
    axis_dim = inp0_betensors.shape[axis]
    positive_bound_mask = indice_betensor >= axis_dim
    indice_betensor[positive_bound_mask] = axis_dim-1
    negative_mask = indice_betensor < 0
    indice_betensor[negative_mask] = indice_betensor[negative_mask] + axis_dim
    negative_bound_mask = indice_betensor < 0
    indice_betensor[negative_bound_mask] = axis_dim-1

    self.outputs[0].betensor = torch.gather(
        inp0_betensors, axis, indice_betensor.long())

    return self.outputs[0].betensor


@quant_register(OpType.GatherElements)
def gather_elements_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
