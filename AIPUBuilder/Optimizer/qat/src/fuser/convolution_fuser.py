# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import sys
import copy
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval
from operator import add, eq, mul

from ..qatregister import register_fusion_pattern
from ..ops import QConvolution2D
from ..utils import extract_conv_hyperparams, replace_node_module


@register_fusion_pattern((nn.Conv2d))
@register_fusion_pattern((nn.BatchNorm2d, nn.Conv2d))
@register_fusion_pattern((nn.PReLU, nn.Conv2d))
@register_fusion_pattern((nn.LeakyReLU, nn.Conv2d))
@register_fusion_pattern((nn.ReLU6, nn.Conv2d))
@register_fusion_pattern((nn.ReLU, nn.Conv2d))
@register_fusion_pattern((nn.PReLU, (nn.BatchNorm2d, nn.Conv2d)))  # 3
@register_fusion_pattern((nn.LeakyReLU, (nn.BatchNorm2d, nn.Conv2d)))  # 2
@register_fusion_pattern((nn.ReLU6, (nn.BatchNorm2d, nn.Conv2d)))  # 1
@register_fusion_pattern((nn.ReLU, (nn.BatchNorm2d, nn.Conv2d)))  # 0, the highest priority
class ConvBNActFusion:
    def __init__(self, quantizer, node):
        self.act_node = None
        self.bn_node = None
        self.conv_node = None
        if isinstance(quantizer.modules[node.target], (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU)):
            self.act_node = node
            prev_node = self.act_node.prev
            if isinstance(quantizer.modules[prev_node.target], (nn.BatchNorm2d)):
                self.bn_node = prev_node
                prev_node = self.bn_node.prev
                if isinstance(quantizer.modules[prev_node.target], (nn.Conv2d)):
                    self.conv_node = prev_node
            elif isinstance(quantizer.modules[prev_node.target], (nn.Conv2d)):
                self.conv_node = prev_node

        elif isinstance(quantizer.modules[node.target], (nn.BatchNorm2d)):
            self.bn_node = node
            prev_node = self.bn_node.prev
            if isinstance(quantizer.modules[prev_node.target], (nn.Conv2d)):
                self.conv_node = prev_node
        elif isinstance(quantizer.modules[node.target], (nn.Conv2d)):
            self.conv_node = node

        self.conv_name = self.conv_node.name
        self.act_function = quantizer.modules[self.act_node.target] if self.act_node else None
        self.bn = quantizer.modules[self.bn_node.target] if self.bn_node else None
        assert self.conv_node is not None, '[FATAL]: conv_node can not be None!'
        self.conv = quantizer.modules[self.conv_node.target]

    def fuse(self, graph_module, modules):
        fused_graph = graph_module.graph
        act_function = self.act_function
        if self.bn_node is not None:
            fused_conv = fuse_conv_bn_eval(self.conv, self.bn)
        else:
            fused_conv = copy.deepcopy(self.conv)

        hyper_params = extract_conv_hyperparams(fused_conv)
        hyper_params['act_function'] = copy.deepcopy(act_function)

        q_conv = QConvolution2D(self.conv_name, **hyper_params, conv_node=fused_conv)
        q_conv.weight.data = fused_conv.weight
        if q_conv.bias is not None:
            q_conv.bias.data = fused_conv.bias

        replace_node_module(self.conv_node, modules, q_conv)

        if self.bn_node is not None:
            replace_node_module(self.bn_node, modules, torch.nn.Identity())
            self.bn_node.replace_all_uses_with(self.conv_node)
            fused_graph.erase_node(self.bn_node)

        if self.act_node is not None:
            replace_node_module(self.act_node, modules, torch.nn.Identity())
            self.act_node.replace_all_uses_with(self.conv_node)
            fused_graph.erase_node(self.act_node)
