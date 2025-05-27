# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.er

import copy
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_linear_bn_eval

from ..qatregister import register_fusion_pattern
from ..ops import QFullyConnected
from ..utils import extract_linear_hyperparams, replace_node_module


@register_fusion_pattern((nn.Linear))
@register_fusion_pattern((nn.BatchNorm1d, nn.Linear))
@register_fusion_pattern((nn.PReLU, nn.Linear))
@register_fusion_pattern((nn.LeakyReLU, nn.Linear))
@register_fusion_pattern((nn.ReLU6, nn.Linear))
@register_fusion_pattern((nn.ReLU, nn.Linear))
@register_fusion_pattern((nn.PReLU, (nn.BatchNorm1d, nn.Linear)))  # 3
@register_fusion_pattern((nn.LeakyReLU, (nn.BatchNorm1d, nn.Linear)))  # 2
@register_fusion_pattern((nn.ReLU6, (nn.BatchNorm1d, nn.Linear)))  # 1
@register_fusion_pattern((nn.ReLU, (nn.BatchNorm1d, nn.Linear)))  # 0, the highest priority
class LinearBNActFusion:
    def __init__(self, quantizer, node):
        self.act_node = None
        self.bn_node = None
        self.linear_node = None
        if isinstance(quantizer.modules[node.target], (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU)):
            self.act_node = node
            prev_node = self.act_node.prev
            if isinstance(quantizer.modules[prev_node.target], (nn.BatchNorm1d)):
                self.bn_node = prev_node
                prev_node = self.bn_node.prev
                if isinstance(quantizer.modules[prev_node.target], (nn.Linear)):
                    self.linear_node = prev_node
            elif isinstance(quantizer.modules[prev_node.target], (nn.Linear)):
                self.linear_node = prev_node

        elif isinstance(quantizer.modules[node.target], (nn.BatchNorm1d)):
            self.bn_node = node
            prev_node = self.bn_node.prev
            if isinstance(quantizer.modules[prev_node.target], (nn.Linear)):
                self.linear_node = prev_node
        elif isinstance(quantizer.modules[node.target], (nn.Linear)):
            self.linear_node = node

        self.fc_name = self.linear_node.name
        self.act_function = quantizer.modules[self.act_node.target] if self.act_node else None
        self.bn = quantizer.modules[self.bn_node.target] if self.bn_node else None
        assert self.linear_node is not None, '[FATAL]: linear_node can not be None!'
        self.linear = quantizer.modules[self.linear_node.target]

    def fuse(self, graph_module, modules):
        fused_graph = graph_module.graph
        act_function = self.act_function
        if self.bn_node is not None:
            fused_linear = fuse_linear_bn_eval(self.linear, self.bn)
        else:
            fused_linear = copy.deepcopy(self.linear)

        hyper_params = extract_linear_hyperparams(fused_linear)
        hyper_params['act_function'] = copy.deepcopy(act_function)

        q_linear = QFullyConnected(self.fc_name, **hyper_params)
        q_linear.weight.data = fused_linear.weight
        if q_linear.bias is not None:
            q_linear.bias.data = fused_linear.bias

        replace_node_module(self.linear_node, modules, q_linear)

        if self.bn_node is not None:
            replace_node_module(self.bn_node, modules, torch.nn.Identity())
            self.bn_node.replace_all_uses_with(self.linear_node)
            fused_graph.erase_node(self.bn_node)

        if self.act_node is not None:
            replace_node_module(self.act_node, modules, torch.nn.Identity())
            self.act_node.replace_all_uses_with(self.linear_node)
            fused_graph.erase_node(self.act_node)
