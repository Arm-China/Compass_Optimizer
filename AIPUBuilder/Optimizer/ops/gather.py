# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *


@op_register(OpType.Gather)
# IR
# layer_bottom=rpn_class/concat_0,topk_indice
# layer_bottom_shape=[1,261888,2],[1,6000]
# layer_bottom_type=float32,float32
# layer_top=gather_prob_tensor
# layer_top_shape=[1,6000,2]
# layer_top_type=float32
# axis=1
# batch_dims=1
# bathc_dims is less than indice'dim, means how many dim as batch
def gather(self, *args):
    inp0_betensors = self.inputs[0].betensor
    # if lut is in IR, one of gather's two inputs is from lut
    # if method =='idx_lut', indice from lut,or gather's object from lut
    if len(self.inputs) < 2:
        lut_idx = self.get_param('method')
        if 'lut' not in self.constants:
            OPT_FATAL("Please check your input or lut should be included in IR")
        else:
            if lut_idx == 'idx_lut':
                indice_betensor = self.constants['lut'].betensor.long()
            else:
                inp0_betensors = self.constants['lut'].betensor
                indice_betensor = self.inputs[0].betensor.long()
    else:
        indice_betensor = self.inputs[1].betensor.long()
    select_valid_index = False
    if indice_betensor.dtype in [Dtype.UINT32, Dtype.INT32, Dtype.INT64, Dtype.UINT64]:
        _, invalid_index = dtype2range(indice_betensor.dtype)
        select_valid_index = True
    # axis = self.get_param('axis')
    # changed_axis = axis
    # if indice_betensor.ndim == 1:
    #     indice_betensor = indice_betensor.unsqueeze(0)
    #     # maskrcnn float IR input shape maybe [...,1]
    #     if inp0_betensors.shape[-1] == 1:
    #         inp0_betensors = inp0_betensors.squeeze()
    #         inp0_betensors = inp0_betensors.unsqueeze(0)
    #         changed_axis = axis + 1
    # if indice_betensor.shape[0] ==  1:
    #     if torch.max(indice_betensor) <= inp0_betensors.shape[changed_axis]:
    #         max_len = inp0_betensors.shape[changed_axis]
    #         pos = self.inputs[1].betensor != max_len
    #         inp_indice = self.inputs[1].betensor[pos].long()
    #     else:
    #         inp_indice = self.inputs[1].betensor[0].long()

    #     inp_betensors = self.inputs[0].betensor

    #     out = torch.index_select(inp_betensors, axis, inp_indice)

    #     if out.shape[axis] < self.outputs[0].shape[axis]:
    #         pad_len = self.outputs[0].shape[axis] - out.shape[axis]
    #         zeros = torch.zeros(list(self.outputs[0].shape),device=inp_betensors.device)
    #         pad_zeros = torch.split(zeros, pad_len, dim=axis)[0]
    #         out_cat = torch.cat((out, pad_zeros), dim=axis)
    #         self.outputs[0].betensor = out_cat
    #     else:
    #         self.outputs[0].betensor = out.reshape(list(self.outputs[0].shape))
    # else: # indice multi dim case
    # out = torch.zeros(self.inputs[1].shape[0], self.inputs[0].shape[1], self.inputs[0].shape[2])
    axis = self.get_param('axis')
    axis = axis if axis >= 0 else len(inp0_betensors.shape) + axis
    batch_dims = self.get_param('batch_dims')
    virtual_batch = torch.prod(torch.tensor(
        inp0_betensors.shape[:batch_dims], device=indice_betensor.device)).int().item()
    # for i in range(indice_betensor.ndim-1):
    #     if indice_betensor.shape[i] == inp0_betensors.shape[i] and i<axis:
    #         virtual_batch = virtual_batch*indice_betensor.shape[i]
    #         batch_dims = batch_dims + 1
    if batch_dims > axis:
        OPT_FATAL("Please check your shape, batch dim should be less than axis ")
    else:
        axis = axis - batch_dims + 1
    newshape = (virtual_batch,) + inp0_betensors.shape[batch_dims:]
    inp0_betensors = inp0_betensors.reshape(newshape)
    # to get total indice length, it maybe be allocated in seveal dims, batch_dims is border
    # such as indice is [10,100,2], if batch_dims is 1, then [100,2] is indice,10 is batch
    outshape = newshape[0:axis] + indice_betensor.shape[batch_dims:] + newshape[axis + 1:]
    temp_out = torch.zeros(outshape, device=indice_betensor.device)
    newshape = (virtual_batch,) + indice_betensor.shape[batch_dims:]

    indice_betensor = indice_betensor.reshape(newshape)
    padding = [0] * (inp0_betensors.dim() - axis) * 2
    padding_value = -self.inputs[0].zerop
    for i in range(virtual_batch):
        # axis include dim, so need axis-1
        if torch.max(indice_betensor) <= inp0_betensors.shape[axis]:
            max_len = inp0_betensors[i].shape[axis - 1]
            pos = indice_betensor[i].flatten() != max_len
            inp_indice = indice_betensor[i].flatten()[pos].long()
        else:
            inp_indice = indice_betensor[i].flatten().long()
        # index_select not support negative index,so we need do some transformation

        neg_index = (inp_indice < 0) * inp0_betensors[i].shape[axis - 1]
        inp_indice = inp_indice + neg_index
        indice_len = valid_num = inp_indice.shape[0]

        # Remove the invalid index
        if select_valid_index:
            mask = torch.where(inp_indice == invalid_index)[0][0]
            inp_indice = inp_indice[:mask]
            valid_num = inp_indice.shape[0]
        out = torch.index_select(inp0_betensors[i], axis - 1, inp_indice)
        padding[-1] = indice_len - valid_num
        out = torch.nn.functional.pad(out, padding, value=padding_value)
        # out shape maybe not match temp_out shape due to some inddex including max index range
        # so only fill actual data to temp_out
        tmp = temp_out[i].flatten()
        tmp[:out.flatten().shape[0]] = out.flatten()
        temp_out[i] = tmp.reshape(temp_out[i].shape)
    self.outputs[0].betensor = temp_out.reshape(self.outputs[0].ir_shape)
    # if out.shape[axis-1] < self.outputs[0].shape[original_axis]:
    #     pad_len = self.outputs[0].shape[original_axis] - out.shape[axis]
    #     shape_zero = list(self.outputs[0].shape)
    #     shape_zero[0] = 1
    #     zeros = torch.zeros(shape_zero,device=inp0_betensors.device)
    #     pad_zeros = torch.split(zeros, pad_len, dim=axis)[0]
    #     out_cat = torch.cat((out, pad_zeros), dim=axis)
    #     self.outputs[0].betensor[i] = out_cat
    # else:
    #     try:
    #         self.outputs[0].betensor[0][i] = out[0]
    #     except:
    #         self.outputs[0].betensor[0][i] = out.reshape(list(self.outputs[0].shape)[1:])

    return self.outputs[0].betensor


@quant_register(OpType.Gather)
def gather_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
