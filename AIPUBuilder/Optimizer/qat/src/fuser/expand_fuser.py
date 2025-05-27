# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import copy
import torch
import torch.nn as nn

from ..qatregister import register_fusion_pattern
from ..qatlogger import QAT_FATAL, QAT_INFO, QAT_ERROR
from ..ops import QExpand
from ..utils import replace_node_module
from types import BuiltinFunctionType


@register_fusion_pattern((torch.Tensor.expand))  # 0, the highest priority
class ExpandFusion:
    def __init__(self, quantizer, node):
        self.expand_node = None
        # if isinstance(node.target, BuiltinFunctionType):
        #     self.expand_node = node
        self.expand_node = node
        assert self.expand_node is not None, '[FATAL]: expand_node can not be None!'

    def fuse(self, graph_module, modules):
        QAT_INFO(f"begin to expand fuse")
        fused_graph = graph_module.graph
        local_args = (self.expand_node.args[0],)
        for arg in self.expand_node.args[1:]:
            if not isinstance(arg, int):
                local_args = self.expand_node.args
                break
        size = [] if len(local_args) > 1 else self.expand_node.args[1:]
        q_expand = QExpand(size=size)
        with fused_graph.inserting_after(self.expand_node):
            graph_module.add_module(self.expand_node.name + "_QExpand", q_expand)
            new_node = fused_graph.call_module(
                self.expand_node.name + "_QExpand", args=local_args)
        self.expand_node.replace_all_uses_with(new_node)
        fused_graph.erase_node(self.expand_node)
