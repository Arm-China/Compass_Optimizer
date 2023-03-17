# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *

register_optype('Repeat')


@op_register(OpType.Repeat)
# IR
# layer_name=mrcnn_detecion/repeat
# layer_type=Repeat
# layer_bottom=[mrcnn_detecion/id_repeats_const_output,mrcnn_detecion/topk_final_sorted_output]
# layer_bottom_shape=[[1,80],[80]]
# layer_bottom_type=[int32,float32]
# layer_top=[mrcnn_detecion/repeat_output]
# layer_top_shape=[[1,1000]]
# layer_top_type=[float32]
# axis=1
def repeat(self, *args):
    # inp_betensors,index = torch.sort(self.inputs[0].betensor,dim=0,descending=False)
    axis = self.get_param("axis", optional=True, default_value=None)
    inp_betensors = self.inputs[0].betensor

    if inp_betensors.ndim > 4:
        OPT_FATAL("repeats dim must be less than 4")
    if axis == None:
        inp_repeats = self.inputs[1].betensor.flatten().long()
        inp_betensors = self.inputs[0].betensor.flatten()
        axis = 0
    else:
        inp_repeats = self.inputs[1].betensor.long()
    # inp_repeats = torch.squeeze(inp_repeats)  # remove size 1's dim
    # out_shape = torch.squeeze(self.outputs[0].betensor).shape #remove size 1's dim
    out_shape = self.outputs[0].ir_shape
    if ((inp_betensors.shape[axis]) != len(inp_repeats)):
        OPT_FATAL("repeats must have the same size as input along axis ")
    if (torch.sum(inp_repeats) > out_shape[axis]):
        OPT_FATAL("repeats number are greater than output size ")
    out = torch.repeat_interleave(inp_betensors, inp_repeats, axis)
    # clone_out = torch.zeros(out_shape,device=self.inputs[0].betensor.device)
    # out shape maybe less than output assigned shape, such as output is 1000,but actual get out result is 500
    # so fill small out to big clone_out
    pad_params = []
    for p in range(out.ndim):
        pad_params.append(0)
        pad_params.append(out_shape[out.ndim-1-p]-out.shape[out.ndim-1-p])
    self.outputs[0].betensor = torch.nn.functional.pad(out, tuple(pad_params), "constant", 0)

    return self.outputs[0].betensor


@quant_register(OpType.Repeat)
def repeat_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
