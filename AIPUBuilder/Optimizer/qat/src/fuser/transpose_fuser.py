# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import copy
import torch
import torch.nn as nn

from ..qatregister import register_fusion_pattern
from ..qatlogger import QAT_FATAL, QAT_INFO, QAT_ERROR
from ..ops import QTranspose
from ..utils import replace_node_module
from types import BuiltinFunctionType


@register_fusion_pattern((torch.permute))  # 0, the highest priority
@register_fusion_pattern((torch.Tensor.permute))  # 0, the highest priority
class TransposeFusion:
    def __init__(self, quantizer, node):
        self.transpose_node = None
        # if isinstance(node.target, BuiltinFunctionType):
        #     self.transpose_node = node
        self.transpose_node = node
        assert self.transpose_node is not None, '[FATAL]: transpose_node can not be None!'

    def fuse(self, graph_module, modules):
        # QAT_INFO(f"begin to transpose fuse")
        fused_graph = graph_module.graph
        local_args = (self.transpose_node.args[0],)
        for arg in self.transpose_node.args[1:]:
            if not isinstance(arg, int):
                local_args = self.transpose_node.args
                break
        perm = [] if len(local_args) > 1 else self.transpose_node.args[1:]
        qname = self.transpose_node.name + "_QTranspose"
        q_transpose = QTranspose(name=qname, perm=perm)
        with fused_graph.inserting_after(self.transpose_node):
            graph_module.add_module(qname, q_transpose)
            new_node = fused_graph.call_module(qname, args=local_args)
        self.transpose_node.replace_all_uses_with(new_node)
        fused_graph.erase_node(self.transpose_node)
