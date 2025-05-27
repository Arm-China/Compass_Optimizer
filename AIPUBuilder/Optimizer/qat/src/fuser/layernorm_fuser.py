# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.er

import copy
import torch
import torch.nn as nn

from ..qatregister import register_fusion_pattern
from ..ops import QLayerNorm
from ..utils import extract_linear_hyperparams, replace_node_module


@register_fusion_pattern((nn.LayerNorm))
class LayerNormFusion:
    def __init__(self, quantizer, node):

        self.ln_node = None
        if isinstance(quantizer.modules[node.target], nn.LayerNorm):
            self.ln_node = node
        self.ln_name = self.ln_node.name
        assert self.ln_node is not None, '[FATAL]: layernorm node can not be None!'
        self.ln_module = quantizer.modules[self.ln_node.target]

    def _extract_hyperparams(self, m, name=None):
        hps = {}
        hps = {
            'name': name,
            'normalized_shape': m.normalized_shape,
            'eps': m.eps,
            'bias': True if hasattr(m, 'bias') and isinstance(m.bias, torch.nn.Parameter) else False
        }
        return hps

    def fuse(self, graph_module, modules):
        hyper_params = self._extract_hyperparams(self.ln_module, self.ln_name)
        qln = QLayerNorm(**hyper_params)
        qln.weight.data = self.ln_module.weight
        if qln.bias is not None:
            qln.bias.data = self.ln_module.bias

        replace_node_module(self.ln_node, modules, qln)
