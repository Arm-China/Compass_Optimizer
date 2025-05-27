# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.er

import copy
import torch
import torch.nn as nn

from ..qatregister import register_fusion_pattern
from ..ops import QFullyConnected, QReshape, QTranspose, QMatMul, QSoftmax, QConcat, QSplit, QBatchNorm, QMultiHeadAttention
from ..qatlogger import QAT_ERROR, QAT_WARN


@register_fusion_pattern((QMultiHeadAttention))
class MHAFusion:
    def __init__(self, quantizer, node):

        self.mha_node = node
        self.mha_module = quantizer.modules[node.target] if node.target in quantizer.modules else None
        if not isinstance(self.mha_module, QMultiHeadAttention):
            self.mha_node = None
            QAT_ERROR(f"multiheadattention fuser failed.")
        self.mha_name = self.mha_node.name

        self.num_heads = self.mha_module.self_attention.num_heads
        self.head_dim = self.mha_module.self_attention.head_dim
        self.embed_dim = self.mha_module.self_attention.embed_dim
        self.kdim = self.mha_module.self_attention.kdim
        self.vdim = self.mha_module.self_attention.vdim

        if hasattr(self.mha_module, 'seq_len') and self.mha_module.seq_len is not None:
            self.seq_len = self.mha_module.seq_len
        else:
            QAT_WARN(f"Seq_len should be setted in MHAFusion by module({self.mha_module}), now using default value=1.")
            self.seq_len = 1

        if not self.mha_module.self_attention._qkv_same_embed_dim:
            self.q_proj_weight = self.mha_module.self_attention.q_proj_weight[:]
            self.k_proj_weight = self.mha_module.self_attention.k_proj_weight[:]
            self.v_proj_weight = self.mha_module.self_attention.v_proj_weight[:]
        else:
            self.q_proj_weight = self.mha_module.self_attention.in_proj_weight[:self.embed_dim, :]
            self.k_proj_weight = self.mha_module.self_attention.in_proj_weight[self.embed_dim:2*self.embed_dim, :]
            self.v_proj_weight = self.mha_module.self_attention.in_proj_weight[2*self.embed_dim:, :]

        if self.mha_module.self_attention.bias_k is not None and self.mha_module.self_attention.bias_v is not None:
            self.bias_k = self.mha_module.self_attention.bias_k[:]
            self.bias_v = self.mha_module.self_attention.bias_v[:]
        else:
            self.bias_k = self.mha_module.self_attention.in_proj_bias[self.embed_dim:2*self.embed_dim]
            self.bias_v = self.mha_module.self_attention.in_proj_bias[2*self.embed_dim:]

        if self.mha_module.self_attention.in_proj_bias is not None:
            self.bias_q = self.mha_module.self_attention.in_proj_bias[:self.embed_dim]
        else:
            self.bias_q = self.mha_module.self_attention.in_proj_bias

        self.out_proj_weight = self.mha_module.self_attention.out_proj.weight.data
        self.out_proj_bias = self.mha_module.self_attention.out_proj.bias.data

        self.need_weights = False
        if 'need_weights' in self.mha_node.kwargs:
            self.need_weights = self.mha_node.kwargs['need_weights']

        if self.need_weights:
            QAT_WARN(f"please attention, now 'need_weights=true.'")

    def fuse(self, graph_module, modules):

        fused_graph = graph_module.graph
        qh_node, kh_node, vh_node = self.mha_node.args
        # with graph_module.graph.inserting_after(self.mha_node):
        with graph_module.graph.inserting_before(self.mha_node):
            qh_fc_name = f"{self.mha_name}_linear_query"
            qfc_node = QFullyConnected(qh_fc_name,
                                       self.embed_dim,
                                       self.embed_dim,
                                       self.bias_q is not None)
            qfc_node.weight.data = self.q_proj_weight
            qfc_node.bias.data = self.bias_q

            graph_module.add_module(qh_fc_name, qfc_node)
            qfc_new_node = fused_graph.call_module(qh_fc_name, (qh_node,))

            kh_fc_name = f"{self.mha_name}_linear_key"
            kfc_node = QFullyConnected(kh_fc_name,
                                       self.embed_dim,
                                       self.kdim,
                                       self.bias_k is not None)
            kfc_node.weight.data = self.k_proj_weight
            kfc_node.bias.data = self.bias_k

            graph_module.add_module(kh_fc_name, kfc_node)
            kfc_new_node = fused_graph.call_module(kh_fc_name, (kh_node,))

            vh_fc_name = f"{self.mha_name}_linear_value"
            vfc_node = QFullyConnected(vh_fc_name,
                                       self.embed_dim,
                                       self.vdim,
                                       self.bias_v is not None)
            vfc_node.weight.data = self.v_proj_weight
            vfc_node.bias.data = self.bias_v

            graph_module.add_module(vh_fc_name, vfc_node)
            vfc_new_node = fused_graph.call_module(vh_fc_name, (vh_node,))

            qreshape_name = f"{self.mha_name}_qreshape"
            qreshape_node = QReshape(shape=[-1, self.seq_len, self.num_heads, self.head_dim], name=qreshape_name)
            graph_module.add_module(qreshape_name, qreshape_node)
            qreshape_new_node = fused_graph.call_module(qreshape_name, (qfc_new_node,))

            kreshape_name = f"{self.mha_name}_kreshape"
            kreshape_node = QReshape(shape=[-1, self.seq_len, self.num_heads, self.head_dim], name=kreshape_name)
            graph_module.add_module(kreshape_name, kreshape_node)
            kreshape_new_node = fused_graph.call_module(kreshape_name, (kfc_new_node,))

            vreshape_name = f"{self.mha_name}_vreshape"
            vreshape_node = QReshape(shape=[-1, self.seq_len, self.num_heads, self.head_dim], name=vreshape_name)
            graph_module.add_module(vreshape_name, vreshape_node)
            vreshape_new_node = fused_graph.call_module(vreshape_name, (vfc_new_node,))

            qtranspose_name = f"{self.mha_name}_qtranspose"
            qtranspose_node = QTranspose(perm=[0, 2, 1, 3], name=qtranspose_name)
            graph_module.add_module(qtranspose_name, qtranspose_node)
            qtranspose_new_node = fused_graph.call_module(qtranspose_name, (qreshape_new_node,))

            ktranspose_name = f"{self.mha_name}_ktranspose"
            ktranspose_node = QTranspose(perm=[0, 2, 3, 1], name=ktranspose_name)
            graph_module.add_module(ktranspose_name, ktranspose_node)
            ktranspose_new_node = fused_graph.call_module(ktranspose_name, (kreshape_new_node,))

            vtranspose_name = f"{self.mha_name}_vtranspose"
            vtranspose_node = QTranspose(perm=[0, 2, 1, 3], name=vtranspose_name)
            graph_module.add_module(vtranspose_name, vtranspose_node)
            vtranspose_new_node = fused_graph.call_module(vtranspose_name, (vreshape_new_node,))

            qk_matmul_name = f"{self.mha_name}_qkmatmul"
            qkmatmul_node = QMatMul(name=qk_matmul_name)
            graph_module.add_module(qk_matmul_name, qkmatmul_node)
            qkmatmul_new_node = fused_graph.call_module(qk_matmul_name, (qtranspose_new_node, ktranspose_new_node,))

            scale_name = f"{self.mha_name}_scale"
            scale_module = QBatchNorm(1, name=scale_name)
            scale_module.weight = torch.rsqrt(torch.tensor(
                self.head_dim, device=self.q_proj_weight.device).reshape([1]))
            graph_module.add_module(scale_name, scale_module)
            scale_node = fused_graph.call_module(scale_name, (qkmatmul_new_node,))
            # use batchnorm is not good, because dynamic shape and quantize, so use constant + mul

            qkmm_softmax_name = f"{self.mha_name}_qkmmsoftmax"
            qkmm_softmax_node = QSoftmax(name=qkmm_softmax_name, dim=-1)
            graph_module.add_module(qkmm_softmax_name, qkmm_softmax_node)
            qksoftmax_new_node = fused_graph.call_module(qkmm_softmax_name, (scale_node,))

            qkv_matmul_name = f"{self.mha_name}_qkvmatmul"
            qkvmatmul_node = QMatMul(name=qkv_matmul_name)
            graph_module.add_module(qkv_matmul_name, qkvmatmul_node)
            qkvmatmul_new_node = fused_graph.call_module(qkv_matmul_name, (qksoftmax_new_node, vtranspose_new_node,))

            oprojpose_name = f"{self.mha_name}_oprojpose"
            oprojpose_node = QTranspose(perm=[0, 2, 1, 3], name=oprojpose_name)
            graph_module.add_module(oprojpose_name, oprojpose_node)
            oprojpose_new_node = fused_graph.call_module(oprojpose_name, (qkvmatmul_new_node,))

            oprojreshape_name = f"{self.mha_name}_oproj_in_reshape"
            oprojreshape_node = QReshape(shape=[-1, self.num_heads * self.head_dim], name=oprojreshape_name)
            graph_module.add_module(oprojreshape_name, oprojreshape_node)
            oprojreshape_new_node = fused_graph.call_module(oprojreshape_name, (oprojpose_new_node,))

            oproj_fc_name = f"{self.mha_name}_linear_outproject"
            oprojfc_module = QFullyConnected(oproj_fc_name,
                                             self.embed_dim,
                                             self.embed_dim,
                                             self.out_proj_bias is not None)
            oprojfc_module.weight.data = self.out_proj_weight
            oprojfc_module.bias.data = self.out_proj_bias

            graph_module.add_module(oproj_fc_name, oprojfc_module)
            oprojfc_new_node = fused_graph.call_module(oproj_fc_name, (oprojreshape_new_node,))

            oprojout_reshape_name = f"{self.mha_name}_oproj_out_reshape"
            oprojout_reshape_node = QReshape(shape=[-1, self.seq_len, self.embed_dim], name=oprojout_reshape_name)
            graph_module.add_module(oprojout_reshape_name, oprojout_reshape_node)
            oprojoutreshape_new_node = fused_graph.call_module(oprojout_reshape_name, (oprojfc_new_node,))

            # oproj_trans_name = f"{self.mha_name}_oproj_out_trans"
            # oproj_trans_module = QTranspose(perm=[0, 1, 2, 3], name=oproj_trans_name)
            # graph_module.add_module(oproj_trans_name, oproj_trans_module)
            # oproj_trans_node  = fused_graph.call_module(oproj_trans_name, (oprojoutreshape_new_node,))

            concat_name = f"{self.mha_name}_concat"
            concat_module = QConcat(dim=-1, name=oprojout_reshape_name)
            graph_module.add_module(concat_name, concat_module)
            # concat_node = fused_graph.call_module(concat_name, ((oproj_trans_node, oproj_trans_node),))
            concat_node = fused_graph.call_module(concat_name, ((oprojoutreshape_new_node, oprojoutreshape_new_node),))

            split_name = f"{self.mha_name}_split"
            split_module = QSplit(split_size_or_sections=self.embed_dim, dim=-1, name=split_name)
            graph_module.add_module(split_name, split_module)
            split_node = fused_graph.call_module(split_name, (concat_node, ))

            self.mha_node.replace_all_uses_with(split_node)
            fused_graph.erase_node(self.mha_node)
