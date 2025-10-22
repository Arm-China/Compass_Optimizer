# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import sys
import copy
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval
from operator import add, eq, mul

from ..qatregister import register_fusion_pattern
from ..ops import QMatMul, QBatchNorm
from ..utils import extract_conv_hyperparams, replace_node_module


@register_fusion_pattern((QBatchNorm, QMatMul))
class MatMulBatchNormFusion:
    def __init__(self, quantizer, node):
        self.bn_node = None
        self.matmul_node = None
        if isinstance(quantizer.modules[node.target], QBatchNorm):
            self.bn_node = node
            prev_node = self.bn_node.prev
            if isinstance(quantizer.modules[prev_node.target], (QMatMul)):
                self.matmul_node = prev_node

            self.matmul_name = self.matmul_node.name
            self.bn_module = quantizer.modules[self.bn_node.target]
            self.matmul_module = quantizer.modules[self.matmul_node.target]

    def fuse(self, graph_module, modules):
        fused_graph = graph_module.graph

        bn_w = self.bn_module.weight
        bn_b = self.bn_module.bias
        if bn_w.unique().numel() == 1 and torch.equal(bn_b, torch.zeros_like(bn_b)):
            fused_multiplier = bn_w.unique().item()
            self.matmul_module.fused_multiplier = fused_multiplier

            replace_node_module(self.bn_node, modules, torch.nn.Identity())
            self.bn_node.replace_all_uses_with(self.matmul_node)
            fused_graph.erase_node(self.bn_node)
