# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
torch_tensor = construct_torch_tensor


def opt_unify_scales_for_multi_inputs_operators(graph, optype_cfg_dt):
    import networkx
    import re

    def is_desired_layer(node):
        n_unquantifiable = node.get_param('unquantifiable', optional=True, default_value=False)
        if node.type == OpType.Quantize:
            return (not node.outputs[0].qinvariant) and ('quantize_scale' in node.params) and n_unquantifiable
        elif OpType.Cast == node.type:
            return False  # (not node.inputs[0].qinvariant) and (node.inputs[0].dtype != node.outputs[0].dtype)
        else:
            return (len(node.outputs) > 0) and (not node.outputs[0].qinvariant) and (not n_unquantifiable) and whether_align_to_out_scale(node)

    def scale2minmax(scale, is_signed, qrange):
        frange = qrange / scale
        if is_signed:
            return -1 * frange / 2,  frange / 2
        else:
            return 0, frange

    net = graph.init_networkx()
    net_reverse = net.reverse(copy=True)

    if not networkx.algorithms.dag.is_directed_acyclic_graph(net):
        OPT_WARN(
            'opt_unify_scales_for_multi_inputs_operators optimization will no be applied, because currently it only support Directed_Acyclic_Graph models')
        return
    graph.set_tensor_quantization_attrs()
    near_knodes = {}
    nscales = {}
    parents_of_operators = {}
    ndepth = {}
    nthreshold = {}
    nmethod = {}
    is_cfg_empty = True if optype_cfg_dt == None else False
    for n in net.nodes:
        n_unquantifiable = n.get_param('unquantifiable', optional=True, default_value=False)
        if is_cfg_empty:
            optype_cfg_dt = {}
            unify_attrs = n.attrs['unify_scales_for_multi_inputs_operators']
            unify_attrs = [x.lower().strip() for x in re.split(r',|\[|\]|\(|\)|\s+',
                                                               unify_attrs.lower().strip()) if x.lower().strip()]
            for idx in range(0, len(unify_attrs),  4):
                if unify_attrs[idx].lower().strip() == n.type.name.lower():
                    optype_cfg_dt[n.type] = [int(unify_attrs[idx+1]), float(unify_attrs[idx+2]), unify_attrs[idx+3]]
                    break
        if n.type in optype_cfg_dt.keys() and len(n.parents) > 1 and not n_unquantifiable and optype_cfg_dt[n.type][2] != 'skip':
            if 'need_align_scales' in n.attrs and not n.attrs['need_align_scales']:
                continue
            inp_dtypes = []
            for inp in n.inputs:
                inp_dtypes.append(inp.dtype)
            if len(set(inp_dtypes)) > 1:
                continue
            cut_off = optype_cfg_dt[n.type][0]
            if cut_off < 1:
                cut_off = None
            near_knodes[n] = []
            # records max, min, avg scale of all key nodes
            nscales[n] = (float('-inf'), float('inf'), 0.0)

            s_count = 0
            traverse_num = 0
            for p in n.parents:
                qinvariant = True
                for t in p.outputs:
                    if t in n.inputs:
                        qinvariant = qinvariant and t.qinvariant
                need_traverse = not (qinvariant or p.get_param('unquantifiable', optional=True, default_value=False))
                if need_traverse:
                    traverse_num += 1
                    tgid = -1
                    branch_knodes = []
                    for anc, d in networkx.single_source_dijkstra_path_length(net_reverse, p,  cutoff=cut_off).items():
                        if (tgid < 0 or tgid == d) and is_desired_layer(anc) and n.attrs['q_bits_activation'] == anc.attrs['q_bits_activation']:
                            if tgid < 0:
                                tgid = d
                            branch_knodes.append(anc)
                            if anc not in parents_of_operators:
                                parents_of_operators[anc] = set([n])
                            else:
                                parents_of_operators[anc].add(n)
                            ascale = anc.outputs[0].scale
                            nscales[n] = (max(nscales[n][0], ascale), min(
                                nscales[n][1], ascale), nscales[n][2] + ascale)
                            s_count += 1
                    if len(branch_knodes) > 0:
                        near_knodes[n].append(branch_knodes)

            nscales[n] = (nscales[n][0], nscales[n][1], nscales[n][2] / max(1.0, float(s_count)))
            nthreshold[n] = optype_cfg_dt[n.type][1]
            nmethod[n] = optype_cfg_dt[n.type][2]
            if len(near_knodes[n]) < traverse_num:
                # did not found enough nodes for each branch in demand
                near_knodes[n] = []

    # merges connected graphs of multiple concats
    # such as  the graph has this structure:  (a0,...)->concat0->b0, (a0,...)->concat1->b1, (a0,...)->concat2->b2
    # so [concat0,concat1,concat2]  has common subinputs, unify the all inputs scales of [Concat0,Concat1,Concat2]
    connected_G = networkx.Graph()
    for n, unify_op in parents_of_operators.items():
        if len(unify_op) > 1:
            num = len(unify_op) - 1
            for idx in range(num):
                connected_G.add_edge(list(unify_op)[idx], list(unify_op)[idx+1])
        else:
            connected_G.add_node(list(unify_op)[0])
    subgraphs = list(networkx.connected_components(connected_G))
    for subgraph in subgraphs:
        num = len(list(subgraph))
        if num == 1:
            continue
        else:
            nodes = list(subgraph)
            merge_op_list = []
            last_op = nodes[0]
            last_attrs = (int(last_op.attrs['layer_id']), last_op.name)
            scales = (float('-inf'), float('inf'), 0.0)
            for n in nodes:
                if n in near_knodes:
                    n_attrs = (int(n.attrs['layer_id']), n.name)
                    if n_attrs > last_attrs:
                        last_op = n
                        last_attrs = n_attrs
                    paths = near_knodes[n]  # paths = [[A,B], [C]]
                    scales = (max(nscales[n][0], scales[0]), min(
                        nscales[n][1], scales[1]), nscales[n][2] + scales[2])
                    merge_op_list.extend(paths)
                    near_knodes.pop(n)
                    nscales.pop(n)
            near_knodes[last_op] = merge_op_list
            nscales[last_op] = (scales[0], scales[1], scales[2] / len(nodes))

    for n, paths in near_knodes.items():
        s_method = nmethod[n]
        n.attrs['unify_scales_for_multi_inputs_operator_threshold'] = nthreshold[n]
        s = 1.0
        if s_method == 'max':
            s = nscales[n][0]
        elif s_method == 'min':
            s = nscales[n][1]
        elif s_method == 'avg':
            s = nscales[n][2]
        else:
            s = n.outputs[0].scale
        for branch in paths:
            for anc in branch:
                t = anc.outputs[0]
                t.scale = s
                t.zerop = torch.zeros_like(t.scale)
                if anc.type == OpType.Quantize:
                    anc.params['quantize_scale'] = s
                    anc.params['quantize_zp'] = 0
                t_min, t_max = scale2minmax(s, is_signed(t.dtype), t.qmax - t.qmin)
                t.min = torch_tensor(t_min, device=n.inputs[0].betensor.device)
                t.max = torch_tensor(t_max, device=n.inputs[0].betensor.device)
                OPT_DEBUG(f"layer_id={anc.attrs['layer_id']}, {str(anc.type)}, {anc.name} : its output0's min/max were changed during opt_unify_scales_for_multi_inputs_operators"
                          f"for layer_id={n.attrs['layer_id']}, {str(n.type)}, {n.name}")
    graph.clear_tensor_quantization_attrs()
