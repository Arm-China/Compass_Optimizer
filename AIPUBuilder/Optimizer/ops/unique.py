# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
import torch

register_optype('Unique')


def find_index(candidate_y, f_value):
    if isinstance(f_value, torch.Tensor):
        for idx, ele in enumerate(candidate_y):
            if (f_value == ele).all():
                return idx
        return -1
    else:
        return candidate_y.index(f_value) if f_value in candidate_y else -1


@op_register(OpType.Unique)
def unique_forward(self, *args):
    need_sort = self.get_param('sorted', optional=True, default_value=True)
    axis = self.get_param('axis', optional=True, default_value=None)
    if isinstance(axis, str):
        if axis.upper() != 'NONE':
            OPT_WARN(f"{self}: axis only support int or None, please check! Now we set axis is None")
        axis = None
    inp = self.inputs[0].betensor
    if need_sort:
        y, inverse_indices, counts = torch.unique(inp, sorted=True, return_inverse=True, return_counts=True, dim=axis)
        inverse_indices = inverse_indices.flatten()
        if axis is None:
            count = y.numel()
        else:
            count = y.shape[axis]
        indices = []
        for i in range(count):
            index = torch.nonzero(inverse_indices == i)[0].item()
            indices.append(index)
        indices = torch.tensor(indices, device=inp.device)
    else:
        from collections import defaultdict
        y = []
        indices = []
        inverse_indices = []
        counts_dict = defaultdict(int)
        if axis is None:
            input_chunk = torch.flatten(inp).tolist()
        else:
            input_chunk = []
            for idx in range(inp.shape[axis]):
                t = torch.index_select(inp, axis, torch.tensor(idx, device=inp.device))
                input_chunk.append(t)
        for idx, ele in enumerate(input_chunk):
            f_index = find_index(y, ele)
            if f_index == -1:
                y.append(ele)
                indices.append(idx)
                f_index = len(y) - 1
            counts_dict[f_index] += 1
            inverse_indices.append(f_index)

        counts = [counts_dict[idx] for idx in range(len(y))]
        if axis is not None:
            y = torch.cat(y, dim=axis)
        else:
            y = torch.tensor(y, device=inp.device)
        indices = torch.tensor(indices, device=inp.device)
        inverse_indices = torch.tensor(inverse_indices, device=inp.device)
        counts = torch.tensor(counts, device=inp.device)

    self.outputs[0].betensor = PyTensor('y', y, self.outputs[0].dtype).betensor
    self.outputs[1].betensor = PyTensor('indices', indices, self.outputs[1].dtype).betensor
    self.outputs[2].betensor = PyTensor('inverse_indices', inverse_indices, self.outputs[2].dtype).betensor
    self.outputs[3].betensor = PyTensor('counts', counts, self.outputs[3].dtype).betensor
    return [o.betensor for o in self.outputs]


@quant_register(OpType.Unique)
def unique_quantize(self, *args):
    inp = self.inputs[0]
    out0 = self.outputs[0]
    out0.dtype = inp.dtype
    out0.scale = inp.scale
    out0.zerop = inp.zerop
    out0.qbits = inp.qbits
    out0.qinvariant = inp.qinvariant
    for out in self.outputs[1:4]:
        out.dtype = out.ir_dtype
        out.qbits = dtype2bits(out.dtype)
        out.scale = 1.0
        out.zerop = 0
        out.qinvariant = True
