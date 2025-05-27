# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import copy
import torch
import torch.nn as nn

from ..qatlogger import QAT_FATAL
from ..qatregister import register_fusion_pattern
from ..ops import QHardSigmoid
from ..utils import replace_node_module


@register_fusion_pattern((nn.Hardsigmoid))
class HardsigmoidFusion:
    def __init__(self, quantizer, node):
        self.hardsigmoid_node = node
        if not isinstance(quantizer.modules[node.target], (nn.Hardsigmoid)):
            self.hardsigmoid_node = None
        if self.hardsigmoid_node is None:
            QAT_FATAL(f"hardsigmoid_node can not be None!")
        self.hardsigmoid_module = quantizer.modules[self.hardsigmoid_node.target]

    def fuse(self, graph_module, modules):
        q_hardsigmoid = QHardSigmoid(name=self.hardsigmoid_node.name)
        replace_node_module(self.hardsigmoid_node, modules, q_hardsigmoid)
