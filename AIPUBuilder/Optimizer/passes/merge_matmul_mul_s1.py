# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
import torch


def merge_constant_mul_to_bn(graph, config=None):
    '''
    Find
    |   Constant
    |       |
    \       /
        Mul
    To
    BatchNorm
    '''
    pending = []
    for node in graph.nodes:
        if node.type != OpType.Mul:
            continue
        if not OpType.Constant in [node.parents[0].type, node.parents[1].type]:
            continue
        if node.parents[0].type == OpType.Constant:
            cnode = node.parents[0]
            lnode = node.parents[1]
        else:
            cnode = node.parents[1]
            lnode = node.parents[0]
        if len(cnode.children) > 1 or cnode.outputs[0] in graph.output_tensors:
            # constant's output has more than one comsumer
            continue
        if cnode.constants['weights'].betensor.flatten().shape[0] > 1:
            continue
        if (cnode.constants['weights'].betensor.ceil() - cnode.constants['weights'].betensor.floor()).max() == 0:
            # batchnorm not support qinvariant operation
            continue

        scale = cnode.constants['weights'].betensor.flatten()[0].item()
        node.remove_input(cnode.outputs[0])
        pending.append(cnode)
        node.type = OpType.BatchNorm
        node.params['axis'] = len(lnode.outputs[0].ir_shape) - 1
        node.params['epsilon'] = 0
        wtshape = TensorShape([lnode.outputs[0].ir_shape[-1]])
        w_t = PyTensor(node.name+"_weight", wtshape)
        w_t.ir_dtype = Dtype.FP32
        w_t.betensor = torch.zeros([node.outputs[0].ir_shape[-1]],
                                   device=cnode.constants['weights'].betensor.device) + scale
        b_t = PyTensor(node.name+"_bias", wtshape)
        b_t.ir_dtype = Dtype.FP32
        b_t.betensor = torch.zeros_like(w_t.betensor)
        node.constants['weights'] = w_t
        node.constants['biases'] = b_t

    for p in pending:
        graph.remove_node(p)


def merge_matmul_mul(graph, config=None):
    # find matmul+scale(represented by operator: bn or mul) and fuse related nodes into Matmul (with additional param: fused_multiplier)
    end_idx = len(graph.nodes)
    node_idx = 0
    ncount = end_idx
    wcount1 = 0
    while node_idx < end_idx and wcount1 < ncount:
        n = graph.nodes[node_idx]
        wcount1 += 1
        if OpType.MatMul == n.type and 1 == len(n.children):
            snodes = []
            flag = False
            fused_multiplier = 1.0
            n1 = n
            n2 = n.children[0]
            wcount2 = 0
            while wcount2 < ncount:
                wcount2 += 1
                if n2.type in OP_ONLY_CHANGE_SHAPE and 1 == len(n2.children):
                    n1 = n2
                    n2 = n2.children[0]
                elif OpType.BatchNorm == n2.type:
                    w = n2.constants['weights'].betensor.unique()
                    b = n2.constants['biases'].betensor.unique()
                    if 1 == w.numel() and 1 == b.numel() and 0. == b.item() and w.item() > 0.:
                        flag = True
                        fused_multiplier = w
                        snodes.append(n2)
                    break
                elif (OpType.Mul == n2.type) or (OpType.Eltwise == n2.type and n2.get_param('method').lower() == 'mul') and n2.get_param('with_activation').lower() == 'none':
                    scale_nodes, count_root, count_constant = (
                        n2.parents[0] if n2.parents[0] != n1 else n2.parents[1]).get_ancestors()
                    if count_root > 0 and count_root == count_constant:
                        gview = graph.subgraph_view(scale_nodes)
                        if len(gview.outflow_tensors) == 1 and gview.outflow_tensors[0] in n2.inputs:
                            for node in scale_nodes:
                                node.forward()
                            s = gview.outflow_tensors[0].betensor.unique()
                            if 1 == s.numel() and s.item() > 0. and n2.outputs[0].ir_shape.size() == (n2.inputs[0] if n2.inputs[1].pnode in scale_nodes else n2.inputs[1]).ir_shape.size():
                                flag = True
                                fused_multiplier = s
                                snodes.extend(gview.nodes)
                                snodes.append(n2)
                    break
                else:
                    break
            for sn in snodes[:-1]:
                for st in (sn.inputs + sn.outputs):
                    if st in (graph.input_tensors + graph.output_tensors):
                        flag = False
                        break
            if flag:
                rn = snodes[-1]
                OPT_DEBUG(f"These nodes will be cut out during pass merge_matmul_mul_s1: {snodes[:-1]}")
                graph.cut_subgraph(snodes[:-1])
                # n.replace_output_temporarily(0, ot)
                if len(rn.inputs) > 1:
                    t = rn.inputs[0] if rn.inputs[0].pnode in snodes else rn.inputs[1]
                    rn.remove_input(t)
                rn.type = OpType.Crop
                rn.params.clear()
                rn.params['crops'] = [[0, i] for i in rn.outputs[0].ir_shape]
                rn.constants.clear()
                rn.placeholders.clear()
                graph.init_networkx()
                n.params['fused_multiplier'] = fused_multiplier
                n.attrs['fused_multiplier'] = fused_multiplier
                node_idx = graph.nodes.index(n)
        node_idx += 1
        end_idx = len(graph.nodes)
    '''
    In addition, find
    BN      BN
    |       |
    NOOP    NOOP
    \\      /
        MM
    if two BN are SAME scaling operations, remove those BN and merge into MM
    '''
    count = 0
    for node in graph.nodes:
        if not node.type == OpType.MatMul:
            continue
        if not len(node.children) == 1:
            continue
        # find 2 bn, using BFS
        pnodes = list(node.parents)
        bn = []
        while len(pnodes) > 0:
            head = pnodes.pop(0)
            if len(head.children) > 1:
                continue
            if head.type == OpType.BatchNorm:
                bn.append(head)
            elif head.type in OP_ONLY_CHANGE_SHAPE:
                pnodes.append(head.parents[0])
            else:
                break
        match = True if len(bn) > 0 else False
        for n in bn:
            if not n.type == OpType.BatchNorm:
                match = False
                break
        if not match:
            continue
        # assert if two bn are scaling operation
        w1 = bn[0].constants['weights'].betensor.unique()
        b1 = bn[0].constants['biases'].betensor.unique()
        if not (1 == w1.numel() and 1 == b1.numel() and 0. == b1.item() and w1.item() > 0.):
            continue
        if len(bn) == 1:
            w2 = 1
            b2 = 0
        else:
            w2 = bn[1].constants['weights'].betensor.unique()
            b2 = bn[1].constants['biases'].betensor.unique()
            if not (1 == w2.numel() and 1 == b2.numel() and 0. == b2.item() and w2.item() > 0.):
                continue
        # Remove two bn and add fuse_multiplier
        node.params['fused_multiplier'] = w1*w2
        node.attrs['fused_multiplier'] = w1*w2
        for rn in bn:
            rn.type = OpType.Crop
            rn.params.clear()
            rn.params['crops'] = [[0, i] for i in rn.outputs[0].ir_shape]
            rn.constants.clear()
            rn.placeholders.clear()
            count += 1
        graph.init_networkx()
    if ncount - len(graph.nodes) + count > 0:
        OPT_INFO(f"{ncount - len(graph.nodes) + count} nodes were cut during pass merge_matmul_mul_s1")


def unmerge_matmul_mul(graph, config=None):
    def condition_func(node, parent_node, edge_tensor):
        return (
            (OpType.MatMul == parent_node.type)
            and parent_node.get_param("unquantifiable", optional=True, default_value=True)
            and ("fused_multiplier" in parent_node.attrs)
        )

    '''
    In LLM like graph, we want to revert mul_scale into Constant+Mul, to adapt dynamic shape feature.
    Otherwise, such as SD graph, we want to keep batchnorm format so that get best performance and reduce
    layoutconvert ops.
    To decide, currently we detect the batch dim of feature map. If it is 2, then we consider it is
    a Cond/Uncond embeded UNet structure.
    '''
    inserted_nodes = graph.insert_dummy_node_ahead(OpType.Mul, condition_func)
    for n in inserted_nodes:
        if n.outputs[0].ir_shape[0] == 1:
            n.params["unquantifiable"] = True
            parent_node = n.parents[0]
            n.attrs.update(parent_node.attrs.clone())
            fused_multiplier = parent_node.attrs["fused_multiplier"]
            if "fused_multiplier" in parent_node.params:
                parent_node.params.pop("fused_multiplier")
            ft = PyTensor(
                graph.get_valid_tensor_name(n.inputs[0].name + "_scale"),
                fused_multiplier,
            )
            ft.dtype = n.inputs[0].dtype
            fn = PyNode(
                graph.get_valid_node_name(parent_node.name + "_scale"), OpType.Constant
            )
            fn.attrs.update(n.attrs.clone())
            fn.constants["weights"] = ft.clone(
                graph.get_valid_tensor_name(ft.name + "_w")
            )
            fn.params["unquantifiable"] = True
            fn.additional = True
            fn.add_output(ft)
            n.add_input(ft)
            graph.add_node(fn)
        else:
            n.type = OpType.BatchNorm
            n.params["unquantifiable"] = True
            parent_node = n.parents[0]
            n.attrs.update(parent_node.attrs.clone())
            fused_multiplier = parent_node.attrs["fused_multiplier"]
            if "fused_multiplier" in parent_node.params:
                parent_node.params.pop("fused_multiplier")
            c = n.outputs[0].ir_shape[-1]
            fw = PyTensor(graph.get_valid_tensor_name(
                n.inputs[0].name + "_scale"), torch.zeros([c]).to(n.inputs[0].betensor.device) + fused_multiplier)
            fb = PyTensor(graph.get_valid_tensor_name(
                n.inputs[0].name + "_bias"), torch.zeros([c]).to(n.inputs[0].betensor.device))
            n.constants['weights'] = fw
            n.constants['biases'] = fb
            n.params['axis'] = len(n.inputs[0].ir_shape) - 1
            n.params['epsilon'] = 0
