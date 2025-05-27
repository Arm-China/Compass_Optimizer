# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
import torch


def detect_inf_mask_nodes(graph, config):
    if config.enable_pass_detect_inf_mask_nodes:
        # filter batchnorm weight and bias's inf,-inf
        for i, n in enumerate(graph.nodes):
            if n.type == OpType.BatchNorm:
                aa = n.constants['weights'].betensor + n.constants['biases'].betensor
                if torch.sum(aa) == 0 and torch.max(n.constants['biases'].betensor) < -65536:
                    n.constants['weights'].betensor = torch.zeros(
                        n.constants['weights'].ir_shape, device=n.constants['weights'].device) + 32767
                    n.constants['biases'].betensor = torch.zeros(
                        n.constants['biases'].ir_shape, device=n.constants['weights'].device) - 32767
            if n.type == OpType.Constant:
                n.constants['weights'].betensor[n.constants['weights'].betensor < -32767] = -32768
