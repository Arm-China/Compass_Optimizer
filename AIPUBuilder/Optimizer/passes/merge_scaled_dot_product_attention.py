# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *

from .merge_matmul_mul_s1 import merge_matmul_mul, unmerge_matmul_mul


def merge_scaled_dot_product_attention(graph, config=None):
    merge_matmul_mul(graph, config)

    end_idx = len(graph.nodes)
    node_idx = 0
    ncount = end_idx
    wcount1 = 0
    while node_idx < end_idx and wcount1 < ncount:
        n = graph.nodes[node_idx]
        wcount1 += 1
        if OpType.Softmax == n.type and n.get_param('axis') in [-1, len(n.inputs[0].ir_shape)-1] and 1 == len(n.children) and OpType.MatMul == n.children[0].type:
            n_pxv = n.children[0]
            edge_v = n_pxv.inputs[1] if OpType.Softmax == n_pxv.inputs[0].pnode.type else n_pxv.inputs[0]
            edge_o = n_pxv.outputs[0]
            edge_m = None
            edge_q = None
            edge_k = None
            np = n.parents[0]
            n_qxk = None
            n_qxk_descendants = []
            for na in sorted(np.get_ancestors()[0], key=lambda nx: nx.attrs['tgid'], reverse=True):
                if OpType.MatMul == na.type and 'fused_multiplier' in na.params:
                    n_qxk = na
                    n_qxk_descendants = n_qxk.get_descendants()[0]
                    break
            snodes = []
            for na in sorted(n_qxk_descendants, key=lambda nx: nx.attrs['tgid'], reverse=False):
                snodes.append(na)
                if na == n_pxv:
                    break
            if ((np.type == OpType.Eltwise and np.get_param('method').lower().strip() == 'add') or np.type == OpType.Add) and (np.get_param('with_activation', optional=True, default_value='none').lower().strip() == 'none') and 1 == len(np.children):
                if np.inputs[0].pnode in snodes:
                    edge_m = np.inputs[1]
                elif np.inputs[1].pnode.type in snodes:
                    edge_m = np.inputs[0]
                else:
                    pass
            flag = False
            if n_qxk:
                flag = True
                edge_q, edge_k = n_qxk.inputs

            for sn in snodes:
                for st in (sn.inputs + sn.outputs):
                    if st in (graph.input_tensors + graph.output_tensors) and st not in [edge_q, edge_k, edge_v, edge_m, edge_o]:
                        flag = False
                        break
            snodes_gv = graph.subgraph_view(snodes)
            for st in snodes_gv.inflow_tensors:
                if st not in [edge_q, edge_k, edge_v, edge_m]:
                    flag = False
                    break
            for st in snodes_gv.outflow_tensors:
                if st not in [edge_o, ]:
                    flag = False
                    break

            if flag:
                an = snodes[-1]
                an.params['trans_v'] = an.get_param('trans_b')
                an.params['trans_k'] = n_qxk.get_param('trans_b')
                an.params['trans_q'] = n_qxk.get_param('trans_a')
                an.params.pop('trans_b')
                an.params.pop('trans_a')
                an.params['enable_gqa'] = True
                an.params['scale_factor'] = n_qxk.get_param('fused_multiplier', optional=True, default_value=1.0)
                an.params['with_mask'] = False
                an.attrs["q_bits_activation"] = 8
                OPT_DEBUG(f"These nodes will be cut out during pass merge_scaled_dot_product_attention: {snodes[:-1]}")
                graph.cut_subgraph(snodes[:-1])
                an.type = OpType.SDPAttention
                an.remove_input(0)
                an.add_input(edge_k, idx=0)
                an.add_input(edge_q, idx=0)
                if edge_m:
                    an.add_input(edge_m, idx=-1)
                    an.params['with_mask'] = True
                an.constants.clear()
                an.placeholders.clear()
                graph.init_networkx()
                node_idx = graph.nodes.index(an)

                # set act per-channel quantization for q, k, v
                def set_sdp_attention_q_mode(nx):
                    nx.attrs['q_mode_activation'] = QuantMode.to_symmetric(
                        QuantMode.to_per_channel(nx.attrs['q_mode_activation']))
                set_sdp_attention_q_mode(an)

                def set_qkv_pnode_q_mode(edge):
                    for pn in sorted(edge.pnode.get_ancestors()[0], key=lambda nx: nx.attrs['tgid'], reverse=True):
                        set_sdp_attention_q_mode(pn)
                        if pn.type not in OP_ONLY_CHANGE_SHAPE:
                            break
                set_qkv_pnode_q_mode(edge_q)
                set_qkv_pnode_q_mode(edge_k)
                set_qkv_pnode_q_mode(edge_v)
                edge_q.name = 'q_' + edge_q.name
                edge_k.name = 'k_' + edge_k.name
                edge_v.name = 'v_' + edge_v.name
        node_idx += 1
        end_idx = len(graph.nodes)

    unmerge_matmul_mul(graph, config)
