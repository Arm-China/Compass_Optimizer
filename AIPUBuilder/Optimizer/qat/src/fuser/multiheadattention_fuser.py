# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import copy
import torch
import torch.nn as nn

from ..qatlogger import QAT_FATAL, QAT_ERROR
from ..qatregister import register_fusion_pattern
from ..ops import QMultiHeadAttention
from ..utils import replace_node_module


@register_fusion_pattern((nn.MultiheadAttention))
class MultiheadAttentionFusion:
    def __init__(self, quantizer, node):
        self.multi_head_attention_node = node
        if not isinstance(quantizer.modules[node.target], (nn.MultiheadAttention)):
            self.multi_head_attention_node = None
        if self.multi_head_attention_node is None:
            QAT_FATAL(f"multi_head_attention_node can not be None!")
        self.attention_module = quantizer.modules[self.multi_head_attention_node.target]

    def _extract_hyperparams(self, m, name=None):
        hp = {
            'embed_dim': m.embed_dim,
            'num_heads': m.num_heads,
            'dropout': m.dropout,
            'bias': m.bias if hasattr(m, 'bias') else False,
            'add_bias_kv': m.add_bias_kv if hasattr(m, 'add_bias_kv') else False,
            'add_zero_attn': m.add_zero_attn,
            'kdim': m.kdim,
            'vdim': m.vdim,
            'batch_first': m.batch_first,
        }
        if name is not None and name != '':
            hp.update({'name': name})
        return hp

    def fuse(self, graph_module, modules):
        hyper_params = self._extract_hyperparams(self.attention_module, name=self.multi_head_attention_node.name)
        if 'need_weights' in self.multi_head_attention_node.kwargs:
            hyper_params.update({'need_weights': self.multi_head_attention_node.kwargs['need_weights']})
        q_multihead_attention = QMultiHeadAttention(**hyper_params)
        if hasattr(self.attention_module, 'in_proj_bias') and self.attention_module.in_proj_bias is not None:
            q_multihead_attention.self_attention.in_proj_bias = copy.deepcopy(self.attention_module.in_proj_bias)
        else:
            QAT_ERROR(f"please support splitted bias for q, k, v")

        if hasattr(self.attention_module, 'in_proj_weight') and self.attention_module.in_proj_weight is not None:
            q_multihead_attention.self_attention.in_proj_weight = copy.deepcopy(self.attention_module.in_proj_weight)
        else:
            QAT_ERROR(f"please support splitted weight for q, k, v")

        q_multihead_attention.self_attention.out_proj.bias = copy.deepcopy(self.attention_module.out_proj.bias)
        q_multihead_attention.self_attention.out_proj.weight = copy.deepcopy(self.attention_module.out_proj.weight)

        replace_node_module(self.multi_head_attention_node, modules, q_multihead_attention)
