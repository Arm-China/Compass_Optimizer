# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import copy
import torch
import torch.nn as nn
from operator import mul, add, eq

from ..qatregister import register_fusion_pattern
from ..qatlogger import QAT_FATAL, QAT_WARN
from ..ops import QElementwiseAdd, QElementwiseMul, QBatchNorm
from ..utils import replace_node_module


@register_fusion_pattern((mul))
@register_fusion_pattern((nn.ReLU, mul))
@register_fusion_pattern((nn.ReLU6, mul))
@register_fusion_pattern((nn.LeakyReLU, mul))
@register_fusion_pattern((nn.PReLU, mul))  # 0, the highest priority
class MulFusion:
    def __init__(self, quantizer, node):
        self.act_node = None
        self.mul_node = None

        if (not eq(node.target, mul)) and isinstance(quantizer.modules[node.target], (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU)):
            self.act_node = node
            prev_node = self.act_node.prev
            if eq(prev_node.target, mul):
                self.mul_node = prev_node
        elif eq(node.target, mul):
            self.mul_node = node

        if self.mul_node is None:
            QAT_FATAL(f"mul_node can not be None!")

        self.act_function = quantizer.modules[self.act_node.target] if self.act_node else None

    def fuse(self, graph_module, modules):
        fused_graph = graph_module.graph
        node0 = self.mul_node._args[0]
        node1 = self.mul_node._args[1]
        act_function = None
        if self.act_node:
            act_function = copy.deepcopy(self.act_function)
        if hasattr(node0, 'op') and hasattr(node1, 'op'):
            if 'get_attr' not in [node0.op, node1.op]:
                mul_name = self.mul_node.name + "_QElementwiseMul"
                q_mul = QElementwiseMul(name=mul_name, act_function=act_function)
                with fused_graph.inserting_after(self.mul_node):
                    graph_module.add_module(mul_name, q_mul)
                    new_node = fused_graph.call_module(mul_name, (node0, node1,))
                self.mul_node.replace_all_uses_with(new_node)
                fused_graph.erase_node(self.mul_node)

                if self.act_node is not None:
                    replace_node_module(self.act_node, modules, torch.nn.Identity())
                    self.act_node.replace_all_uses_with(new_node)
                    fused_graph.erase_node(self.act_node)
            else:
                '''one of the two ops is get_attr'''
                if 'get_attr' in [node0.op]:
                    input_node = node1
                    fused_w = node0.target
                else:
                    input_node = node0
                    fused_w = node1.target
                if len(fused_w.shape) <= 1:  # scalar or vector
                    mul_name = self.mul_node.name + "_QBatchNorm"
                    q_mul = QBatchNorm(fused_w.numel(), name=mul_name)
                    q_mul.weight.data = fused_w
                    q_mul.bias.data = torch.zeros_like(fused_w)
                    with fused_graph.inserting_after(self.mul_node):
                        graph_module.add_module(mul_name, q_mul)
                        new_node = fused_graph.call_module(mul_name, (input_node, ))
                    self.mul_node.replace_all_uses_with(new_node)
                    fused_graph.erase_node(self.mul_node)
        else:
            if hasattr(node0, 'op'):
                input_node = node0
                fused_w = node1
            else:
                input_node = node1
                fused_w = node0

            fused_w = torch.tensor(fused_w)
            if len(fused_w.shape) <= 1:  # scalar or vector
                mul_name = self.mul_node.name + "_QBatchNorm"
                q_mul = QBatchNorm(fused_w.numel(), name=mul_name)
                q_mul.weight.data = fused_w
                q_mul.bias.data = torch.zeros_like(fused_w)
                with fused_graph.inserting_after(self.mul_node):
                    graph_module.add_module(mul_name, q_mul)
                    new_node = fused_graph.call_module(mul_name, (input_node, ))
                self.mul_node.replace_all_uses_with(new_node)
                fused_graph.erase_node(self.mul_node)


@register_fusion_pattern((add))
@register_fusion_pattern((nn.ReLU, add))
@register_fusion_pattern((nn.ReLU6, add))
@register_fusion_pattern((nn.LeakyReLU, add))
@register_fusion_pattern((nn.PReLU, add))  # 0, the highest priority
class AddFusion:
    def __init__(self, quantizer, node):
        self.act_node = None
        self.add_node = None

        if (not eq(node.target, add)) and isinstance(quantizer.modules[node.target], (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU)):
            self.act_node = node
            prev_node = self.act_node.prev
            if eq(prev_node.target, add):
                self.add_node = prev_node
        elif eq(node.target, add):
            self.add_node = node
        assert self.add_node is not None, '[FATAL]: add_node can not be None!'

        self.act_function = quantizer.modules[self.act_node.target] if self.act_node else None

    def fuse(self, graph_module, modules):
        fused_graph = graph_module.graph
        node0 = self.add_node._args[0]
        node1 = self.add_node._args[1]
        act_function = None
        if self.act_node:
            act_function = copy.deepcopy(self.act_function)

        if 'get_attr' in [node0.op, node1.op]:
            QAT_WARN(f"{self.add_node} has constant input")

        add_name = self.add_node.name + "_QElementwiseAdd"
        q_add = QElementwiseAdd(name=add_name, act_function=act_function)
        with fused_graph.inserting_after(self.add_node):
            graph_module.add_module(add_name, q_add)
            new_node = fused_graph.call_module(add_name, (node0, node1,))
        self.add_node.replace_all_uses_with(new_node)
        fused_graph.erase_node(self.add_node)

        if self.act_node is not None:
            replace_node_module(self.act_node, modules, torch.nn.Identity())
            self.act_node.replace_all_uses_with(new_node)
            fused_graph.erase_node(self.act_node)
