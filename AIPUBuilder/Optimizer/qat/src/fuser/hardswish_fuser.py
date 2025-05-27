# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import copy
import torch
import torch.nn as nn

from ..qatlogger import QAT_FATAL
from ..qatregister import register_fusion_pattern
from ..ops import QHardSwish
from ..utils import replace_node_module


@register_fusion_pattern((nn.Hardswish))
class HardswishFusion:
    def __init__(self, quantizer, node):
        self.hardswish_node = node
        if not isinstance(quantizer.modules[node.target], (nn.Hardswish)):
            self.hardswish_node = None
        if self.hardswish_node is None:
            QAT_FATAL(f"hardswish_node can not be None!")
        self.hardswish_module = quantizer.modules[self.hardswish_node.target]

    def fuse(self, graph_module, modules):
        q_hardswish = QHardSwish(name=self.hardswish_node.name)
        replace_node_module(self.hardswish_node, modules, q_hardswish)
