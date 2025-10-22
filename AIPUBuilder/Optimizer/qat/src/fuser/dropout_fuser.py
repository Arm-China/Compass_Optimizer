# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch

from ..qatregister import register_fusion_pattern
from ..ops import QDropOut
from ..qatlogger import QAT_INFO


@register_fusion_pattern((torch.nn.Dropout))
class DropOutFusion:
    def __init__(self, quantizer, node):
        self.dropout_node = node
        self.name = node.name

    def fuse(self, graph_module, modules):
        QAT_INFO(f"find dropout and fuse to Qdropout")
        fused_graph = graph_module.graph
        q_dropout = QDropOut(name=self.name)
        new_module_name = self.dropout_node.name + "_QDropout"
        with fused_graph.inserting_after(self.dropout_node):
            graph_module.add_module(new_module_name, q_dropout)
            new_qdropout_module = fused_graph.call_module(new_module_name, args=self.dropout_node.args)
        self.dropout_node.replace_all_uses_with(new_qdropout_module)
        fused_graph.erase_node(self.dropout_node)
