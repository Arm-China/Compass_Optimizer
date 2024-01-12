# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *


@op_register(OpType.StridedSlice)
def stridedslice(self, *args):
    x = self.inputs[0].betensor

    batch_size = x.shape[0]
    ir_batch_size = self.inputs[0].ir_shape[0]
    start_data_idx = self.current_batch_idx * batch_size

    strides = self.get_param('strides')
    begin = self.get_param('begin')
    end = self.get_param('end')
    upper_bond = self.get_param('upper_bound', optional=True, default_value=False)
    input_shape = self.inputs[0].ir_shape
    for i in range(len(begin)):
        begin[i] = begin[i] + input_shape[i] if begin[i] < 0 else begin[i]
    for i in range(len(end)):
        end[i] = end[i] + input_shape[i]*(strides[i] > 0) + upper_bond if end[i] < 0 else end[i]

    for axis in range(len(strides)):
        start_index = begin[axis]
        end_index = end[axis]
        stride = strides[axis]
        step = int((abs(end_index - start_index) - 1) / abs(stride))
        actual_index = [start_index + stride * s for s in range(step+1)]
        if axis == 0:  # process single or multiply batch
            batch_index = actual_index
            actual_index = []
            inorder_batch_size_idx = range(0, batch_size, 1) if end_index > start_index else range(batch_size-1, -1, -1)
            for batch_idx in inorder_batch_size_idx:
                current_batch_idx = (start_data_idx + batch_idx) % ir_batch_size
                if current_batch_idx in batch_index:
                    actual_index.append(batch_idx)
        x = x.index_select(axis, torch.tensor(actual_index, device=x.device))
    self.outputs[0].betensor = x
    return x


@quant_register(OpType.StridedSlice)
def stridedslice_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qmin, out.qmax = inp.qmin, inp.qmax
    out.qinvariant = inp.qinvariant

    if out.is_perchannel_quantization():
        strides = self.get_param('strides')
        begin = self.get_param('begin')
        end = self.get_param('end')
        upper_bond = self.get_param('upper_bound', optional=True, default_value=False)

        b = begin[out.key_axis]
        e = end[out.key_axis]
        s = strides[out.key_axis]
        index = torch.arange(b, e, s, device=inp.device)

        out.scale = inp.scale.index_select(0, index)
        out.zerop = inp.zerop.index_select(0, index)
