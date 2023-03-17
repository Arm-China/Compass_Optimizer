# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *


def singleGatherNd(inp_betensors, inp_indice):
    # input data last dim is keeped, if input dim == indice dim, input's shape[-1] as 1
    # indice's last dim indicate how many input's dim need be indiced
    newshape = inp_indice.shape[:-1]+inp_betensors.shape[inp_indice.shape[-1]:]
    out = torch.zeros(newshape, device=inp_betensors.device)
    if out.ndim == 1:
        out = out.unsqueeze(-1)
    if inp_indice.ndim > 2:
        m = 1
        for i in range(inp_indice.ndim-1):
            m = m*inp_indice.shape[i]
        inp_indice = inp_indice.reshape(m, inp_indice.shape[-1])
        newshape = (m,)+inp_betensors.shape[inp_indice.shape[-1]:]
        out = torch.zeros(newshape, device=inp_betensors.device)

    if(inp_betensors.ndim == inp_indice.shape[-1]):
        inp_betensors = inp_betensors.unsqueeze(-1)
    if inp_indice.shape[-1] == 1:
        for seq in range(newshape[0]):
            out[seq, ...] = inp_betensors[inp_indice[seq, :][0], ...]
    elif inp_indice.shape[-1] == 2:
        for seq in range(newshape[0]):
            out[seq, ...] = inp_betensors[inp_indice[seq, :][0], inp_indice[seq, :][1], ...]
    elif inp_indice.shape[-1] == 3:
        for seq in range(newshape[0]):
            out[seq, ...] = inp_betensors[inp_indice[seq, :][0], inp_indice[seq, :][1], inp_indice[seq, :][2], ...]
    elif inp_indice.shape[-1] == 4:
        for seq in range(newshape[0]):
            out[seq, ...] = inp_betensors[inp_indice[seq, :][0], inp_indice[seq, :]
                                          [1], inp_indice[seq, :][2], inp_indice[seq, :][3], ...]
    elif inp_indice.shape[-1] == 5:
        for seq in range(newshape[0]):
            out[seq, ...] = inp_betensors[inp_indice[seq, :][0], inp_indice[seq, :][1],
                                          inp_indice[seq, :][2], inp_indice[seq, :][3], inp_indice[seq, :][4], ...]
    else:
        OPT_FATAL("need indice data dim only is 2 or 3,4,5 ")
    return out


register_optype('GatherND')


@op_register(OpType.GatherND)
# IR
# layer_type=GatherND
# layer_bottom=[mrcnn_detecion/reshape_deltas_output,mrcnn_detecion/concat_gathernd_index_output]
# layer_bottom_shape=[[1000,81,4],[1000,2]]
# layer_bottom_type=[float32,float32]
# layer_top=[mrcnn_detecion/gather_delta_output]
# layer_top_shape=[[1000,4]]
# layer_top_type=[float32]
###########################################
# layer_bottom_shape=[[2,458,274,1],[916,3]]
# layer_top=[BatchGatherND/GatherNd_0]
# layer_top_shape=[[916,1]]
# batch_dims=0
# indice shape[-1]<=input's rank
# inp=[[[1,2],[[3,4]],[[5,6]],[[7,8]]],shape is [2,2,2]
# dice=[[[1,0]],[[0,1]]],shape is [2,1,2]
# batch_dim is 0,
def gatherND(self, *args):

    inp_betensors = self.inputs[0].betensor
    inp_indice = (self.inputs[1].betensor).long()

    batch_dims = self.get_param('batch_dims', optional=True, default_value=0)
    batchnum = inp_betensors.shape[0]

    if batch_dims == 0:
        # input data last dim is keeped, if input dim == indice dim, input's shape[-1] as 1
        # indice's last dim indicate how many input's dim need be indiced
        if inp_indice.ndim == 1:
            inp_indice = inp_indice.unsqueeze(0)
        newshape = inp_indice.shape[:-1]+inp_betensors.shape[inp_indice.shape[-1]:]
        out = singleGatherNd(inp_betensors, inp_indice)
    else:  # batch dim==1
        newshape = inp_indice.shape[0:1]+inp_indice[0, ...].shape[:-1] + \
            inp_betensors[0, ...].shape[inp_indice.shape[-1]:]
        out = torch.zeros(newshape, device=inp_betensors.device)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        for batch in range(batchnum):
            indice_batch = inp_indice[batch, ...]
            inp_batch = inp_betensors[batch, ...]
            if indice_batch.ndim == 1:
                indice_batch = indice_batch.unsqueeze(0)
            out[batch, ...] = singleGatherNd(inp_batch, indice_batch).reshape(out[batch, ...].shape)
    self.outputs[0].betensor = out.reshape(self.outputs[0].ir_shape)

    return self.outputs[0].betensor


@quant_register(OpType.GatherND)
def gatherND_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
