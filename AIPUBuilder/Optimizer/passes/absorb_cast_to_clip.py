# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import OpType, Dtype
from AIPUBuilder.Optimizer.utils import dtype2torch_type, dtype2range, is_float
import torch


def criteria(n):
    if n.type != OpType.Activation or n.params['method'].lower() != 'clip':
        return False

    if n.inputs[0].dtype != Dtype.INT32 or n.outputs[0].dtype != Dtype.INT32:
        return False

    if not n.get_param('unquantifiable', optional=True, default_value=False):
        return False

    if len(n.parents) != 1 or len(n.children) != 1:
        return False

    if n.parents[0].type != OpType.Cast or n.children[0].type != OpType.Cast:
        return False

    if not is_float(n.parents[0].inputs[0].dtype) or not is_float(n.children[0].outputs[0].dtype):
        return False

    return True


def absorb_cast_to_clip(graph, config=None):
    """
    merge the patten:  fp16-->cast-->int32 --> clip-->int32-->cast-->fp16 to fp16-->clip-->fp16, and clip the clip_max to fp16 max.
    """
    for n in graph.nodes:
        if not criteria(n):
            continue

        cast_p = n.parents[0]
        cast_c = n.children[0]

        clip_new_inp = cast_p.inputs[0]
        clip_new_out = cast_c.outputs[0]

        n.remove_input(0)
        n.add_input(clip_new_inp)
        n.remove_output(0)
        n.add_output(clip_new_out)

        dtype = clip_new_out.dtype
        if is_float(dtype):
            dtype_max = torch.finfo(dtype2torch_type(dtype)).max
            # dtype_min = torch.finfo(dtype2torch_type(dtype)).min
        else:
            _, dtype_max = dtype2range(dtype)

        n.params['clip_max'] = torch.clamp_max(torch.tensor(n.params['clip_max']), dtype_max).item()
        graph.remove_node(cast_p)
        graph.remove_node(cast_c)
