# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import math

# IR
# layer_type=TopKV2
# layer_bottom=reshape_out
# layer_bottom_shape=[1,261888]
# layer_bottom_type=float32
# layer_top=topk_prob,topk_indice
# layer_top_shape=[1,6000],[1,6000]
# layer_top_type=float32,float32
# k=6000
# axis
# select_index


def reorder_index(origin_order):
    values = origin_order[0]
    index = origin_order[1]
    axis_dim = values.shape[-1]
    all_dim = values.numel()
    length = all_dim//axis_dim
    new_shape = (length, axis_dim)
    new_values = values.reshape(new_shape)
    new_index = index.reshape(new_shape)
    idx = 0
    while idx < length:
        tmp_values = new_values[idx, :]
        tmp_index = new_index[idx, :]
        l = 0
        while l < tmp_values.shape[0]:
            equal_value = torch.where(tmp_values == tmp_values[l])[0]  # tmp_values == tmp_values[l]
            equal_len = len(equal_value)
            if equal_len > 1:
                equal_index = tmp_index[l:l+equal_len]
                equal_index = torch.flip(equal_index, dims=[0])
                tmp_index[l:l+equal_len] = equal_index
                l += equal_len
            else:
                l += 1
        new_index[idx, :] = tmp_index
        idx += 1
    new_values = torch.reshape(new_values, values.shape)
    new_index = torch.reshape(new_index, values.shape)
    return [new_values, new_index]


@op_register(OpType.TopK)
def topk(self, *args):
    if len(self.inputs) > 1:
        k = self.inputs[1].betensor.item()
    else:
        k = self.get_param('k')
    k = int(k)
    axis = self.get_param('axis')
    k = min(self.inputs[0].betensor.shape[axis], k)
    largest = self.get_param('largest', optional=True, default_value=True)
    issorted = self.get_param('sorted', optional=True, default_value=True)
    axis = axis if axis >= 0 else self.outputs[0].betensor.ndim+axis

    if 'select_index' not in self.params:
        if 'extra_params' not in self.attrs:
            select_index = 'last'
        else:
            extra_params = self.attrs['extra_params']
            smethod = int(extra_params[1] if len(extra_params) > 1 else -1)
            select_index = 'last' if smethod < 0 else 'first'
        self.params['select_index'] = select_index
    else:
        select_index = self.get_param('select_index')

    out1 = torch.zeros_like(self.outputs[0].betensor, dtype=torch.float64)
    out2 = torch.zeros_like(self.outputs[1].betensor, dtype=torch.int64)
    inp_betensors = self.inputs[0].betensor

    if select_index == 'random':
        [out1, out2] = torch.topk(inp_betensors, k, axis, largest, issorted)
    elif select_index == 'first':
        transp_inp = torch.transpose(inp_betensors, axis, -1)
        out_order = torch.sort(transp_inp, dim=-1, descending=largest, stable=True)
        transp_out = out_order[0][..., 0:k], out_order[1][..., 0:k]
        recover_out0 = torch.transpose(transp_out[0], axis, -1)
        recover_out1 = torch.transpose(transp_out[1], axis, -1)

        [out1, out2] = recover_out0, recover_out1
    else:  # last mode
        transp_inp = torch.transpose(inp_betensors, axis, -1)
        out_order = torch.sort(transp_inp, dim=-1, descending=largest, stable=True)
        out_order = reorder_index(out_order)
        transp_out = out_order[0][..., 0:k], out_order[1][..., 0:k]

        recover_out0 = torch.transpose(transp_out[0], axis, -1)
        recover_out1 = torch.transpose(transp_out[1], axis, -1)

        [out1, out2] = recover_out0, recover_out1

    self.outputs[0].betensor = out1
    self.outputs[1].betensor = out2
    return [out1, out2]


@quant_register(OpType.TopK)
def topk_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]

    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    # for indice no quant
    out = self.outputs[1]
    out.scale = 1.0
    out.zerop = 0
    axis = self.get_param('axis')
    index_max = inp.ir_shape[axis]
    out.qbits = min(32, max(16, math.log2(index_max)))
    out.dtype = bits2dtype(out.qbits, self.force_dtype_int)
    out.qinvariant = True

    if 'select_index' not in self.params:
        extra_params = self.attrs['extra_params']
        smethod = int(extra_params[1] if len(extra_params) > 1 else -1)
        self.params['select_index'] = 'last' if smethod < 0 else 'first'
