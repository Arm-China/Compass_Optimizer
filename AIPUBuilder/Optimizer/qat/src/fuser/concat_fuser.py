# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import copy
import torch
import torch.nn as nn
from operator import eq

from ..qatregister import register_fusion_pattern
from ..qatlogger import QAT_WARN
from ..ops import QConcat
from ..utils import replace_node_module


@register_fusion_pattern((torch.cat))
@register_fusion_pattern((torch.concat))
class ConcatFusion:
    def __init__(self, quantizer, node):
        self.cat_node = None
        if eq(node.target, torch.concat) or eq(node.target, torch.cat):
            self.cat_node = node
        assert self.cat_node is not None, '[FATAL]: cat_node can not be None!'
        if 'dim' in self.cat_node.kwargs:
            self.dim = self.cat_node.kwargs['dim']
        elif len(self.cat_node.args) > 1:
            self.dim = self.cat_node.args[1]
        else:
            self.dim = -1
            QAT_WARN(f"not found the dim parameters in cat node, using dim=-1.")

    def fuse(self, graph_module, modules):
        qname = self.cat_node.name + "_QConcat"
        q_cat = QConcat(dim=self.dim, name=qname)
        fused_graph = graph_module.graph
        with fused_graph.inserting_after(self.cat_node):
            graph_module.add_module(qname, q_cat)
            new_node = fused_graph.call_module(qname, args=self.cat_node.args)
        self.cat_node.replace_all_uses_with(new_node)
        fused_graph.erase_node(self.cat_node)
