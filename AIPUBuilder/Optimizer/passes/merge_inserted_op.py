# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import OpType
from AIPUBuilder.Optimizer.utils import OP_ONLY_CHANGE_SHAPE, dtype2bits


def _quantize_cast_criteria(n, stricted_condition=True):
    if n.type in [OpType.Quantize]:
        children = n.children
        """
                   /--> cast -->
        1. quantize --|--> cast -->   or  2. quantize --> cast -->
                   \--> cast -->
        now only support 2: only one consumer condition.
        """
        if len(children) == 1:
            visited_nodes = [n, children[0]]
            visiting_node = children[0]
            while visiting_node:
                if visiting_node.type == OpType.Cast:
                    visited_nodes.append(visiting_node)
                    break
                if visiting_node.type not in OP_ONLY_CHANGE_SHAPE:
                    break
                if len(visiting_node.children) > 1 or len(visiting_node.children) == 0:
                    # this will exclude the slice/split which in OP_ONLY_CHANGE_SHAPE
                    break

                visited_nodes.append(visiting_node)
                visiting_node = visiting_node.children[0]
            if visited_nodes[-1].type == OpType.Cast:
                if stricted_condition:
                    cast_n = visited_nodes[-1]
                    out_bits = dtype2bits(cast_n.outputs[0].dtype)
                    in_bits = dtype2bits(cast_n.inputs[0].dtype)
                    if out_bits >= in_bits:
                        return True, visited_nodes
                else:
                    return True, visited_nodes
    return False, []


def _cast_dequantize_criteria(n):
    if n.type in [OpType.DeQuantize]:
        parents = n.parents
        """
                   /--> dequantize -->
        1. cast  --|--> dequantize -->
                   \--> dequantize -->
        2. cast --> dequantize -->
        3. cast --> reshape --> dequantize -->
        now only support 2 and 3: only one consumer condition.
        """
        if len(parents) == 1:
            visited_nodes = [n, parents[0]]
            visiting_node = parents[0]
            while visiting_node:
                if visiting_node.type == OpType.Cast:
                    visited_nodes.append(visiting_node)
                    break
                if visiting_node.type not in OP_ONLY_CHANGE_SHAPE:
                    break
                if visiting_node.type in [OpType.Slice, OpType.Split]:
                    # this will exclude the slice/split which in OP_ONLY_CHANGE_SHAPE
                    break
                if len(visiting_node.parents) > 1 or len(visiting_node.parents) == 0:
                    break

                visited_nodes.append(visiting_node)
                visiting_node = visiting_node.parents[0]
            if visited_nodes[-1].type == OpType.Cast:
                return True, visited_nodes
    return False, []


def _merge_quantize_relative_op(graph, config):

    def _merge_quantize_cast(g, visited_node):
        quantize_n = visited_node[0]
        cast_n = visited_node[-1]
        for vn in visited_node[:-1]:
            for ot in vn.outputs:
                ot.clone_qinfo(cast_n.outputs[0])

        cast_n.type = OpType.NoOp
        cast_n.params.clear()
        cast_n.params['original_type'] = 'OpType.Cast'

        quantize_n.params["quantize_scale"] = quantize_n.outputs[0].scale
        quantize_n.params["quantize_zp"] = quantize_n.outputs[0].zerop

    # firstly, merge quantize + cast(out_bits >= in_bits)
    for n in graph.nodes:
        matched, visited_nodes = _quantize_cast_criteria(n)
        if matched and len(visited_nodes):
            _merge_quantize_cast(graph, visited_nodes)


def _merge_dequantize_relative_op(graph, config):

    def _merge_dequantize_cast(qg, visited_node):
        dequantize_n = visited_node[0]
        cast_n = visited_node[-1]
        for vn in visited_node[1:]:
            for ot in vn.outputs:
                ot.clone_qinfo(cast_n.inputs[0])

        cast_n.type = OpType.NoOp
        cast_n.params.clear()
        cast_n.params['original_type'] = 'OpType.Cast'

        dequantize_n.params["quantize_scale"] = dequantize_n.inputs[0].scale
        dequantize_n.params["quantize_zp"] = dequantize_n.inputs[0].zerop

    # firstly, merge quantize + cast
    for n in graph.nodes:
        matched, visited_node = _cast_dequantize_criteria(n)
        if matched and len(visited_node):
            _merge_dequantize_cast(graph, visited_node)


def merge_inserted_op(graph, config=None):
    _merge_quantize_relative_op(graph, config)
    _merge_dequantize_relative_op(graph, config)
