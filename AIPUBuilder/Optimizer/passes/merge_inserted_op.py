# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import OpType, PyNode
from AIPUBuilder.Optimizer.utils import OP_ONLY_CHANGE_SHAPE, dtype2bits
from AIPUBuilder.Optimizer.logger import OPT_WARN, OPT_INFO


def in_tensor_consumers(n, t):
    consumers = []
    t_producers = t.pnode
    if t.pnode is not None:
        for pc in t_producers.children:
            if t in pc.inputs:
                consumers.append(pc)
    return consumers


def quantize_cast_criteria(n, stricted_condition=True):
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


def cast_dequantize_criteria(n):
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


def merge_quantize_or_dequantize_criteria(n, type):
    flag = False
    if n.type in [type]:
        consumers = in_tensor_consumers(n, n.inputs[0])
        if len(consumers) > 1:
            flag = True
            for consumer in consumers:
                """
                            / --> quantize -->
                1. tensor --| --> quantize -->
                            \ ....

                            / --> reshape --> quantize -->
                2. tensor --| --> quantize -->
                            \ ....

                2 now does not support now.

                """
                if consumer.type != type:
                    flag = False
                    break
                if not consumer.outputs[0].is_qinfo_equal(n.outputs[0]):
                    flag = False
                    break
            if flag:
                return flag, consumers
    return flag, []


def merge_quantize_relative_op(graph, fgraph, config=None):
    fgraph_n_name_2_n = {}
    for fn in fgraph.nodes:
        fgraph_n_name_2_n.update({fn.name: fn})

    def _merge_quantize_cast(g, visited_node, fgraph, fgraph_n_name_2_n):
        quantize_n = visited_node[0]
        cast_n = visited_node[-1]
        for vn in visited_node[:-1]:
            for ot in vn.outputs:
                ot.clone_qinfo(cast_n.outputs[0])

        for cn in cast_n.children:
            idx = cn.remove_input(cast_n.outputs[0])
            cn.add_input(cast_n.inputs[0], idx)

        f_cast_n = fgraph_n_name_2_n[cast_n.name]
        for cn in f_cast_n.children:
            idx = cn.remove_input(f_cast_n.outputs[0])
            cn.add_input(f_cast_n.inputs[0], idx)

        g.remove_node(cast_n)
        fgraph.remove_node(f_cast_n)

        quantize_n.params["quantize_scale"] = quantize_n.outputs[0].scale
        quantize_n.params["quantize_zp"] = quantize_n.outputs[0].zerop

    # firstly, merge quantize + cast(out_bits >= in_bits)
    for n in graph.nodes:
        matched, visited_nodes = quantize_cast_criteria(n)
        if matched and len(visited_nodes):
            _merge_quantize_cast(graph, visited_nodes, fgraph, fgraph_n_name_2_n)

    # secondly, merge quantizes which have same tensor
    def _multi_branch_merge(merged_nodes, quantize_n, g):
        for mn in merged_nodes:
            children = mn.children
            for cn in children:
                idx = cn.remove_input(mn.outputs[0])
                cn.add_input(quantize_n.outputs[0], idx)
        for mn in merged_nodes:
            g.remove_node(mn)

    for n in graph.nodes:
        matched, merged_node = merge_quantize_or_dequantize_criteria(n, OpType.Quantize)
        if matched and len(merged_node):
            merged_node.remove(n)
            _multi_branch_merge(merged_node, n, graph)

            f_merged_nodes = [fgraph_n_name_2_n[mn.name] for mn in merged_node]
            fn = fgraph_n_name_2_n[n.name]
            _multi_branch_merge(f_merged_nodes, fn, fgraph)

    # last, merge quantize + cast(out_bits < in_bits)
    for n in graph.nodes:
        matched, visited_nodes = quantize_cast_criteria(n, stricted_condition=False)
        if matched and len(visited_nodes):
            _merge_quantize_cast(graph, visited_nodes, fgraph, fgraph_n_name_2_n)


def merge_dequantize_relative_op(graph, fgraph, config=None):
    fgraph_n_name_2_n = {}
    for fn in fgraph.nodes:
        fgraph_n_name_2_n.update({fn.name: fn})

    def _merge_dequantize_cast(qg, visited_node, fg, fgraph_n_name_2_n):
        def _remove_cast(cast_node, g):
            children = cast_node.children
            for cn in children:
                idx = cn.remove_input(cast_node.outputs[0])
                cn.add_input(cast_node.inputs[0], idx)
            g.remove_node(cast_node)

        dequantize_n = visited_node[0]
        cast_n = visited_node[-1]
        for vn in visited_node[1:]:
            for ot in vn.outputs:
                ot.clone_qinfo(cast_n.inputs[0])
        _remove_cast(cast_n, qg)
        _remove_cast(fgraph_n_name_2_n[cast_n.name], fg)

        dequantize_n.params["quantize_scale"] = dequantize_n.inputs[0].scale
        dequantize_n.params["quantize_zp"] = dequantize_n.inputs[0].zerop

    # firstly, merge quantize + cast
    for n in graph.nodes:
        matched, visited_node = cast_dequantize_criteria(n)
        if matched and len(visited_node):
            _merge_dequantize_cast(graph, visited_node, fgraph, fgraph_n_name_2_n)

    # secondly, merge dequantizes which have same tensor
    def _multi_branch_merge(merged_nodes, dequantize_n, g):
        for mn in merged_nodes:
            children = mn.children
            for cn in children:
                idx = cn.remove_input(mn.outputs[0])
                cn.add_input(dequantize_n.outputs[0], idx)
            g.remove_node(mn)

    for n in graph.nodes:
        matched, merged_node = merge_quantize_or_dequantize_criteria(
            n, OpType.DeQuantize
        )
        if matched and len(merged_node):
            merged_node.remove(n)

            _multi_branch_merge(merged_node, n, graph)

            f_merged_nodes = [fgraph_n_name_2_n[mn.name] for mn in merged_node]
            fn = fgraph_n_name_2_n[n.name]
            _multi_branch_merge(f_merged_nodes, fn, fgraph)


def merge_quantize_dequantize(graph, fgraph, config=None):
    fgraph_n_name_2_n = {}
    for fn in fgraph.nodes:
        fgraph_n_name_2_n.update({fn.name: fn})

    remove_nodes = []
    for n in graph.nodes:
        if (
            n.type == OpType.Quantize
            and len(n.children) == 1
            and n.children[0].type == OpType.DeQuantize
        ):
            if len(n.children[0].children) == 0:
                # if dequantize has no child, we donot merge the quantize + dequantize, because we want to reserve the
                # output tensor in graph
                continue
            in_tensor = n.inputs[0]
            ot_tensor = n.children[0].outputs[0]
            for cn in n.children[0].children:
                idx = cn.remove_input(ot_tensor)
                cn.add_input(in_tensor, idx)

            fn = fgraph_n_name_2_n[n.name]
            for fcn in fn.children[0].children:
                idx = fcn.remove_input(fn.children[0].outputs[0])
                fcn.add_input(fn.inputs[0], idx)

            remove_nodes.append(n)
            remove_nodes.append(n.children[0])
        if (
            n.type == OpType.DeQuantize
            and len(n.children) == 1
            and n.children[0].type == OpType.Quantize
        ):
            if len(n.children[0].children) == 0:
                # if quantize has no child, we donot merge the dequantize + quantize, because we want to reserve the
                # output tensor in graph
                continue
            in_tensor = n.inputs[0]
            ot_tensor = n.children[0].outputs[0]
            in_bits = dtype2bits(in_tensor.dtype)
            out_bits = dtype2bits(ot_tensor.dtype)
            if in_bits > out_bits:
                OPT_WARN(
                    f"when merge dequantize and quantize, the input qbits(={in_bits}) of dequantize is "
                    f"larger than output qbits(={out_bits}) of quantize, which may cause acc issue and may not"
                    f"meet the lib spec because the input dtype(={in_tensor.dtype}) of dequantize changes to "
                    f"the output dtype(={ot_tensor.dtype}) of quantize."
                )
            for cn in n.children[0].children:
                idx = cn.remove_input(ot_tensor)
                cn.add_input(in_tensor, idx)

            fn = fgraph_n_name_2_n[n.name]
            for fcn in fn.children[0].children:
                idx = fcn.remove_input(fn.children[0].outputs[0])
                fcn.add_input(fn.inputs[0], idx)

            remove_nodes.append(n)
            remove_nodes.append(n.children[0])

    for rn in remove_nodes:
        graph.remove_node(rn)
        fn = fgraph_n_name_2_n[rn.name]
        fgraph.remove_node(fn)


def _swap_n(one_n, is_float=False):
    changed_node = None
    parent_node = one_n.parents[0]
    in_tensor_parent_node = parent_node.inputs[0]
    ot_tensor_quantize_node = one_n.outputs[0]

    _ = one_n.remove_input(0)
    _ = one_n.remove_output(0)
    one_n.add_input(in_tensor_parent_node)

    _ = parent_node.remove_input(0)
    _ = parent_node.remove_output(0)
    parent_node.add_output(ot_tensor_quantize_node)

    qn_output = in_tensor_parent_node.clone()
    qn_output.clone_qinfo(ot_tensor_quantize_node)
    parent_node.add_input(qn_output)
    if not is_float:
        changed_node = parent_node
    one_n.add_output(qn_output)

    return changed_node


def lift_quantize(qgraph, fgraph, config=None):
    def _lift(g, fg):
        fgraph_n_name_2_n = {}
        for fn in fg.nodes:
            fgraph_n_name_2_n.update({fn.name: fn})

        is_lifted = False
        changed_node = []
        for n in g.nodes:
            if (
                n.type in [OpType.Quantize]
                and len(n.parents) == 1
                and n.parents[0].type in OP_ONLY_CHANGE_SHAPE
                and len(n.parents[0].children) == 1
            ):
                if n.parents[0].type in [OpType.Slice, OpType.Split, OpType.Repeat]:
                    continue
                if len(n.parents[0].parents) > 1:
                    continue

                fn = fgraph_n_name_2_n[n.name]
                changed_node.append(_swap_n(n))
                if fn != n:
                    _swap_n(fn, True)
                is_lifted = True
        for cn in changed_node:
            if isinstance(cn, PyNode):
                cn.unquantifiable = False
        return is_lifted

    graph_changed = True
    while graph_changed:
        graph_changed = _lift(qgraph, fgraph)


def merge_inserted_op(graph, fgraph, config=None):
    """
    this pass is quantize(or dequantize) + cast merge, quantize lift, dequantize lower and quantize + dequantize merge

    when quantize + cast merge:
        1. if cast is lower_bits < out_bits, directly merge quantize + cast
        2. if merge multi-quantize to one quantize which from same tensor
        3. lift quantize
    """
    OPT_INFO(f"begin to merge inserted op")
    merge_quantize_relative_op(graph, fgraph, config)
    merge_dequantize_relative_op(graph, fgraph, config)
    lift_quantize(graph, fgraph, config)
    merge_quantize_dequantize(graph, fgraph, config)
