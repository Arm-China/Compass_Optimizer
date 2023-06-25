# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *


def opt_unify_scales_for_multi_inputs_operators(graph, optype_cfg_dt):
    import networkx

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
    for n in net.nodes:
        n_unquantifiable = n.get_param('unquantifiable', optional=True, default_value=False)
        if n.type in optype_cfg_dt.keys() and len(n.parents) > 1 and not n_unquantifiable:
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
                        if (tgid < 0 or tgid == d) and (anc.type != n.type) and is_desired_layer(anc) and n.attrs['q_bits_activation'] == anc.attrs['q_bits_activation']:
                            if tgid < 0:
                                tgid = d
                            branch_knodes.append(anc)
                            ascale = anc.outputs[0].scale
                            nscales[n] = (max(nscales[n][0], ascale), min(
                                nscales[n][1], ascale), nscales[n][2] + ascale)
                            s_count += 1
                    if len(branch_knodes) > 0:
                        near_knodes[n].append(branch_knodes)
            nscales[n] = (nscales[n][0], nscales[n][1], nscales[n][2] / max(1.0, float(s_count)))
            if len(near_knodes[n]) < traverse_num:
                # did not found enough nodes for each branch in demand
                near_knodes[n] = []
    for n, paths in near_knodes.items():
        s_method = optype_cfg_dt[n.type][-1].lower().strip()
        n.attrs['unify_scales_for_multi_inputs_operator_threshold'] = optype_cfg_dt[n.type][1]
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
                t.zerop = 0
                if anc.type == OpType.Quantize:
                    anc.params['quantize_scale'] = s
                    anc.params['quantize_zp'] = 0
                t.min, t.max = scale2minmax(s, is_signed(t.dtype), t.qmax - t.qmin)
                OPT_DEBUG(f"layer_id={anc.attrs['layer_id']}, {str(anc.type)}, {anc.name} : its output0's min/max were changed during opt_unify_scales_for_multi_inputs_operators"
                          f"for layer_id={n.attrs['layer_id']}, {str(n.type)}, {n.name}")
    graph.clear_tensor_quantization_attrs()
