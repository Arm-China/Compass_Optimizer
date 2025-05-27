# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import copy
import torch
import torch.nn as nn

from ..qatlogger import QAT_FATAL
from ..qatregister import register_fusion_pattern
from ..ops import QAveragePooling2D, QMaxPooling2D
from ..utils import replace_node_module


@register_fusion_pattern((nn.AdaptiveAvgPool2d))
@register_fusion_pattern((nn.AvgPool2d))  # 0, the highest priority
class AvgPool2dFusion:
    def __init__(self, quantizer, node):
        self.avg_node = None

        if isinstance(quantizer.modules[node.target], (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            self.avg_node = node

        if self.avg_node is None:
            QAT_FATAL(f"avg_node can not be None!")

        self.avg_module = quantizer.modules[self.avg_node.target]
        self.avg_name = self.avg_node.name

    def fuse(self, graph_module, modules):
        avg_module = copy.deepcopy(self.avg_module)
        q_avg = QAveragePooling2D(avg_module, self.avg_name)
        replace_node_module(self.avg_node, modules, q_avg)


@register_fusion_pattern((nn.AdaptiveMaxPool2d))
@register_fusion_pattern((nn.MaxPool2d))  # 0, the highest priority
class MaxPool2dFusion:
    def __init__(self, quantizer, node):
        self.max_node = None

        if isinstance(quantizer.modules[node.target], (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            self.max_node = node

        assert self.max_node is not None, '[FATAL]: max_node can not be None!'

        self.max_module = quantizer.modules[self.max_node.target]
        self.max_name = self.max_node.name

    def fuse(self, graph_module, modules):
        max_module = copy.deepcopy(self.max_module)
        q_max = QMaxPooling2D(max_module, self.max_name)
        replace_node_module(self.max_node, modules, q_max)
