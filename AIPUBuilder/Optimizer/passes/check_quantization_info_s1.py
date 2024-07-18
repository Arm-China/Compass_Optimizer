# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *


def check_quantization_info(graph: PyGraph, config=None):
    """
    check the 16bits quantization should be symmetric
    :param graph:
    :param config:
    :return:
    """
    for node in graph.nodes:
        activation_mode = node.attrs.get('q_mode_activation')
        activation_bits = node.attrs.get('q_bits_activation')
        if activation_bits >= 16 and QuantMode.is_asymmetric(activation_mode):
            sym_activation_mode = QuantMode.to_symmetric(activation_mode)
            node.attrs['q_mode_activation'] = sym_activation_mode
            OPT_DEBUG(f"{node} changes quantization method of activation tensor "
                      f"from {activation_mode} to {sym_activation_mode}")
