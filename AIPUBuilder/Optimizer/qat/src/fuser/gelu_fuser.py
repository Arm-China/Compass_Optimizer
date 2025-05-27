# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import copy
import torch
import torch.nn as nn

from ..qatlogger import QAT_FATAL
from ..qatregister import register_fusion_pattern
from ..ops import QGeLU
from ..utils import replace_node_module


@register_fusion_pattern((nn.GELU))
class GeLUFusion:
    def __init__(self, quantizer, node):
        self.gelu_node = node
        if not isinstance(quantizer.modules[node.target], (nn.GELU)):
            self.gelu_node = None
        if self.gelu_node is None:
            QAT_FATAL(f"gelu_node can not be None!")
        self.gelu_module = quantizer.modules[self.gelu_node.target]

    def fuse(self, graph_module, modules):
        q_gelu = QGeLU(name=self.gelu_node.name, approximate=self.gelu_module.approximate)
        replace_node_module(self.gelu_node, modules, q_gelu)
