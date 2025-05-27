# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.de

from AIPUBuilder.Optimizer.framework import OpType
from AIPUBuilder.Optimizer.logger import OPT_WARN, OPT_DEBUG


ELIMINATE_OPS = {}


def eliminate_op_register(func):
    if func.__name__ not in ELIMINATE_OPS:
        ELIMINATE_OPS[func.__name__] = func

    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
    return wrapper


def _remove_one_node(g, n):
    if len(n.inputs) != 1 or len(n.outputs) != 1:
        OPT_WARN(f"only remove single-input and single-output")
        return

    in_t = n.inputs[0]
    ot_t = n.outputs[0]
    n_in_is_graph_in_tensor = in_t in g.input_tensors
    n_in_is_graph_ot_tensor = in_t in g.output_tensors
    n_ot_is_graph_ot_tensor = ot_t in g.output_tensors

    graph_change = False
    if not n_ot_is_graph_ot_tensor:
        for cn in n.children:
            idx = cn.remove_input(ot_t)
            cn.add_input(in_t, idx)
            graph_change = True
    elif not n_in_is_graph_in_tensor and not n_in_is_graph_ot_tensor and n_ot_is_graph_ot_tensor:
        pn = n.parents[0]  # single inputs has only one parent
        pnchildren = pn.children
        idx = pn.remove_output(in_t)
        pn.add_output(ot_t, idx)

        for cn in pnchildren:
            if cn == n:
                continue
            if in_t in cn.inputs:
                idx = cn.remove_input(in_t)
                cn.add_input(ot_t, idx)
        graph_change = True

    if graph_change:
        n.remove_input(0)
        n.remove_output(0)
        g.remove_node(n)


@eliminate_op_register
def _eliminate_noop(graph, config):
    def _criteria(node):
        return True if node.type == OpType.NoOp else False

    for n in graph.nodes:
        if not _criteria(n):
            continue
        _remove_one_node(graph, n)

    for n in graph.nodes:
        if n.type == OpType.NoOp:
            n.type = OpType.Reshape
            for k, _ in n.params.items():
                n.params.pop(k)
            n.params['shape'] = list(n.outputs[0].ir_shape)


@eliminate_op_register
def _eliminate_cast(graph, config):
    def _criteria(node):
        if not (node.type == OpType.Cast and hasattr(node, 'additional') and node.additional):
            return False
        to_dtype = node.params['to_dtype']
        in_dtype = node.inputs[0].dtype

        if 'only_for_quantized' in node.params:
            node.params.pop('only_for_quantized')

        if to_dtype != in_dtype:
            return False

        if 'scale_value' in node.params and 'shift_value' in node.params:
            if not n.inputs[0].is_qinfo_equal(n.outputs[0]):
                return False
            ts = node.params['scale_value'] * (0.5 ** node.params['shift_value'])
            if ts != 1.0:
                return False
        return True

    for n in graph.nodes:
        if not _criteria(n):
            continue
        _remove_one_node(graph, n)


@eliminate_op_register
def _elminate_reshape(graph, config):
    def _criteria(n):
        if n.type != OpType.Reshape:
            return False
        in_shape = list(n.inputs[0].ir_shape)
        ot_shape = list(n.outputs[0].ir_shape)
        shape = list(n.params['shape'])
        if not(in_shape == ot_shape == shape and n.inputs[0].is_qinfo_equal(n.outputs[0])):
            return False
        return True

    for n in graph.nodes:
        if not _criteria(n):
            continue
        _remove_one_node(graph, n)


def eliminate_ops(graph, config):
    for k, func in ELIMINATE_OPS.items():
        OPT_DEBUG(f"begin to eliminate pass:{k}")
        func(graph, config)


if __name__ == '__main__':
    print(ELIMINATE_OPS)

    for k, func in ELIMINATE_OPS.items():
        print(k)
