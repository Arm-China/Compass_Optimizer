# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *


def check_16bits_activation_quant_mode(node):
    activation_mode = node.attrs.get('q_mode_activation')
    activation_bits = node.attrs.get('q_bits_activation')
    if activation_bits >= 16 and QuantMode.is_asymmetric(activation_mode):
        sym_activation_mode = QuantMode.to_symmetric(activation_mode)
        node.attrs['q_mode_activation'] = sym_activation_mode
        OPT_DEBUG(f"{node} changes quantization method of activation tensor "
                  f"from {activation_mode} to {sym_activation_mode}")


def check_quantization_info(graph: PyGraph, config=None):
    """
    check the 16bits quantization should be symmetric
    :param graph:
    :param config:
    :return:
    """
    for node in graph.nodes:
        quant_type = node.attrs.get('quant_type')
        if quant_type == 'disable':
            check_16bits_activation_quant_mode(node)
        elif node.type not in QuantType.fp_quant_op_type(quant_type):
            node.attrs['quant_type'] = 'disable'
            check_16bits_activation_quant_mode(node)
        else:
            activation_bits = QuantType.activation_bits(quant_type)
            if activation_bits != node.attrs.get('q_bits_activation'):
                OPT_DEBUG(f"{node} changes q_bits_activation "
                          f"from {node.attrs.get('q_bits_activation')} to {activation_bits}")
                node.attrs['q_bits_activation'] = activation_bits

            weight_bits = QuantType.weight_bits(quant_type)
            if weight_bits != node.attrs.get('q_bits_weight'):
                OPT_DEBUG(f"{node} changes q_bits_weight "
                          f"from {node.attrs.get('q_bits_weight')} to {weight_bits}")
                node.attrs['q_bits_weight'] = weight_bits

            bias_bits = QuantType.bias_bits(quant_type)
            if bias_bits != node.attrs.get('q_bits_bias'):
                OPT_DEBUG(f"{node} changes q_bits_bias "
                          f"from {node.attrs.get('q_bits_bias')} to {bias_bits}")
                node.attrs['q_bits_bias'] = bias_bits

            if is_float(QuantType._to_Dtype(QuantType.activation_type(quant_type))) and QuantMode.is_asymmetric(node.attrs.get('q_mode_activation')):
                sym_mode = QuantMode.to_symmetric(node.attrs.get('q_mode_activation'))
                OPT_DEBUG(f"{node} changes quantization method of activation tensor "
                          f"from {node.attrs.get('q_mode_activation')} to {sym_mode}")
                node.attrs['q_mode_activation'] = sym_mode
            if is_float(QuantType._to_Dtype(QuantType.weight_type(quant_type))) and QuantMode.is_asymmetric(node.attrs.get('q_mode_weight')):
                sym_mode = QuantMode.to_symmetric(node.attrs.get('q_mode_weight'))
                OPT_DEBUG(f"{node} changes quantization method of weight tensor "
                          f"from {node.attrs.get('q_mode_weight')} to {sym_mode}")
                node.attrs['q_mode_weight'] = sym_mode
            if is_float(QuantType._to_Dtype(QuantType.bias_type(quant_type))) and QuantMode.is_asymmetric(node.attrs.get('q_mode_bias')):
                sym_mode = QuantMode.to_symmetric(node.attrs.get('q_mode_bias'))
                OPT_DEBUG(f"{node} changes quantization method of bias tensor "
                          f"from {node.attrs.get('q_mode_bias')} to {sym_mode}")
                node.attrs['q_mode_bias'] = sym_mode
            if not is_float(QuantType._to_Dtype(QuantType.activation_type(quant_type))):
                check_16bits_activation_quant_mode(node)
