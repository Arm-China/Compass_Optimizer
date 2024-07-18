# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *


def decompose_nonmonotonic_activations(graph: PyGraph, config=None):
    nodes_removed = []
    for node in graph.nodes:
        if not node.type == OpType.Activation or not node.params['method'].upper() in ['GELU', 'SILU', 'HARDSWISH', 'SWISH', 'MISH'] or not config.enable_pass_decompose_nonmonotonic_activations.get(node):
            continue
        feature_map_size = node.outputs[0].ir_shape.size()
        if feature_map_size < 2 * 1024 * 1024:  # less than 2MB
            continue
        nodes_removed.append(node)
        output = node.outputs[0]
        inp = node.inputs[0]

        s_n = PyNode(graph.get_valid_node_name(node.name + "/nonmonotonic"), OpType.Activation)
        s_n.params['method'] = "UNKNOWN"
        s_n.attrs = node.attrs.clone()
        s_n.attrs['lut_items_in_bits'] = 8  # for target <= X2, lut size in aiff is 256
        if node.params['method'].upper() == 'GELU':
            from AIPUBuilder.Optimizer.ops.gelu import phi_tanh_approx
            s_n.attrs['lambda_func'] = lambda x: phi_tanh_approx(x)
            s_n.attrs['out_signed'] = False
            OPT_INFO(f"Decomposing GELU act node ({node}) into X mul Φ(x)")
        elif node.params['method'].upper() == 'SILU':
            s_n.attrs['lambda_func'] = lambda x: torch.nn.functional.sigmoid(x)
            s_n.attrs['out_signed'] = False
            OPT_INFO(f"Decomposing SILU act node ({node}) into X mul sigmoid(x)")
        elif node.params['method'].upper() == 'SWISH':
            s_n.attrs['lambda_func'] = lambda x: torch.nn.functional.sigmoid(node.get_param('alpha') * x)
            s_n.attrs['out_signed'] = False
            OPT_INFO(f"Decomposing SWISH act node ({node}) into X mul sigmoid(alpha * x)")
        elif node.params['method'].upper() == 'HARDSWISH':
            s_n.attrs['lambda_func'] = lambda x: torch.nn.functional.hardsigmoid(x)
            s_n.attrs['out_signed'] = False
            OPT_INFO(f"Decomposing HARDSWISH act node ({node}) into X mul hardsigmoid(x)")
        elif node.params['method'].upper() == 'MISH':
            s_n.attrs['lambda_func'] = lambda x: torch.nn.functional.softplus(x).tanh()
            s_n.attrs['out_signed'] = False
            OPT_INFO(f"Decomposing MISH act node ({node}) into X mul Tanh(Softplus(x))")

        s_t = output.clone(graph.get_valid_tensor_name(s_n.name + "_0"))
        s_n.add_output(s_t)

        mul_n = PyNode(graph.get_valid_node_name(node.name + "/Mul"), OpType.Mul)
        mul_n.attrs = node.attrs.clone()

        s_n.add_input(inp)
        mul_n.add_input(inp)
        mul_n.add_input(s_t)
        mul_n.add_output(output)

        graph.add_node(s_n)
        graph.add_node(mul_n)

    for node in nodes_removed:
        graph.remove_node(node)
