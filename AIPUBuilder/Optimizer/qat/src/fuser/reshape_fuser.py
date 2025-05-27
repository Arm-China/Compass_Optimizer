# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import copy
import torch
import torch.nn as nn

from ..qatregister import register_fusion_pattern
from ..qatlogger import QAT_FATAL, QAT_INFO, QAT_ERROR
from ..ops import QReshape
from ..utils import replace_node_module
from types import BuiltinFunctionType


@register_fusion_pattern((torch.flatten))
@register_fusion_pattern((torch.reshape))  # 0, the highest priority
@register_fusion_pattern((torch.Tensor.reshape))  # 0, the highest priority
class ReshapeFusion:
    def __init__(self, quantizer, node):
        self.reshape_node = None
        self._is_flatten = False

        if isinstance(node.target, BuiltinFunctionType):
            self.reshape_node = node
            self._is_flatten = True
        if '_QReshape' in node.name:
            self._is_flatten = False

        self.name = node.name

        assert self.reshape_node is not None, '[FATAL]: reshape_node can not be None!'

    def fuse(self, graph_module, modules):
        # QAT_INFO(f"begin to reshape fuse")
        q_reshape = None
        fused_graph = graph_module.graph
        if self._is_flatten:
            if len(self.reshape_node.args) == 1:
                start_dim, end_dim = 0, -1
            elif len(self.reshape_node.args) == 2:
                start_dim, end_dim = self.reshape_node.args[1], -1
            elif len(self.reshape_node.args) == 3:
                start_dim, end_dim = self.reshape_node.args[1:]
            else:
                start_dim, end_dim = 0, -1
                QAT_ERROR(f"please check the reshape args len")
            q_reshape = QReshape(start_dim=start_dim, end_dim=end_dim, name=self.name)
        else:
            shapes = self.reshape_node.args[1:]
            q_reshape = QReshape(shape=shapes, name=self.name)
        # else:
        #     shape = self.reshape_node.shape
        #     q_reshape = QReshape(shape = shape)

        with fused_graph.inserting_after(self.reshape_node):
            graph_module.add_module(self.reshape_node.name + "_QReshape", q_reshape)
            new_node = fused_graph.call_module(
                self.reshape_node.name + "_QReshape", args=self.reshape_node.args)
        self.reshape_node.replace_all_uses_with(new_node)
        fused_graph.erase_node(self.reshape_node)
