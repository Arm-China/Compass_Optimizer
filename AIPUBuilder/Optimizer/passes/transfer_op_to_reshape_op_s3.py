# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *


def criteria(n):
    ret = False
    if n.type == OpType.Cast:
        if (n.parents[0].attrs['q_mode_activation'] == n.attrs['q_mode_activation'] and
            len(n.inputs) > 0 and len(n.outputs) > 0 and
                n.inputs[0].dtype == n.outputs[0].dtype):
            ret = True
    elif n.type == OpType.FakeQuantWithMinMaxVars:
        ret = True
    else:
        pass
    return ret


def transfer_op_to_reshape_op(g, config):
    # transform useless op to lightweight reshape op
    need_replace_ops = []
    for n in g.nodes:
        if n is not None and criteria(n):
            # create reshape node
            transform_op = PyNode(n.name, OpType.Reshape)
            transform_op.additional = True
            # set attrs and params
            transform_op.attrs.update(n.attrs.clone())
            transform_op.params['shape'] = n.outputs[0].ir_shape
            # record pairs
            need_replace_ops.append((n, transform_op))
    for old, new in need_replace_ops:
        g.replace_node_safely(old, new)
