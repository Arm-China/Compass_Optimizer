# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from typing import Union
from torch import nn
from .common_utils import convert2tuple
from ..qatlogger import QAT_ERROR


def extract_conv_hyperparams(m: nn.Conv2d):
    hyper_params = {
        'in_channels': m.in_channels,
        'out_channels': m.out_channels,
        'kernel_size': convert2tuple(m.kernel_size),
        'stride': convert2tuple(m.stride),
        'padding': convert2tuple(m.padding),
        'dilation': convert2tuple(m.dilation),
        'groups': m.groups,
        'bias': m.bias is not None,
        'padding_mode': m.padding_mode,
    }
    return hyper_params


def extract_avgpool_hyperparams(m: nn.AvgPool2d):
    hyper_params = {
        'kernel_size': m.kernel_size,
        'stride': m.stride,
        'padding': convert2tuple(m.padding),
        'ceil_mode': m.ceil_mode,
        'count_include_pad': m.count_include_pad,
        'divisor_override': m.divisor_override
    }
    return hyper_params


def convert_adaptive2avg_pool_hyperparams(m: nn.AdaptiveAvgPool2d, input_shape):
    in_H, in_W = input_shape[2], input_shape[3]
    if isinstance(m.output_size, (list, tuple)) and len(m.output_size) == 2:
        out_H, out_W = m.output_size
    elif isinstance(m.output_size, int):
        out_H, out_W = m.output_size, m.output_size
    else:
        QAT_ERROR(f"m.output_size(={m.output_size}) is not list, tuple or int, please help to check")
        out_H, out_W = 1, 1

    assert out_H*out_W == 1, '[FATAL]: for AdaptiveAvgPool2d, only support output_size (1, 1)'

    hyper_params = {
        'kernel_size': (in_H, in_W),
        'stride': (1, 1),
        'padding': (0, 0),
        'ceil_mode': False,
        'count_include_pad': False,
        'divisor_override': None,
    }
    return hyper_params


def extract_maxpool_hyperparams(m: nn.MaxPool2d):
    hyper_params = {
        'kernel_size': m.kernel_size,
        'stride': m.stride,
        'padding': convert2tuple(m.padding),
        'dilation': convert2tuple(m.dilation),
        'return_indices': m.return_indices,
        'ceil_mode': m.ceil_mode,
    }
    return hyper_params


def convert_adaptive2max_pool_hyperparams(m: nn.AdaptiveMaxPool2d, input_shape):
    in_H, in_W = input_shape[2], input_shape[3]
    out_H, out_W = m.output_size

    assert out_H*out_W == 1, '[FATAL]: for AdaptiveMaxPool2d, only support output_size (1, 1)'

    hyper_params = {
        'kernel_size': (in_H, in_W),
        'stride': (1, 1),
        'padding': (0, 0),
        'dilation': (1, 1),
        'return_indices': False,
        'ceil_mode': False,
    }
    return hyper_params


def extract_pad_hyperparams(pad_node):
    hyper_params = {
        'pad': pad_node.kwargs['pad'],
        'mode': pad_node.kwargs['mode'],
        'value': pad_node.kwargs['value'],
    }
    return hyper_params


def extract_linear_hyperparams(linear: nn.Linear):
    hyper_params = {
        'in_features': linear.in_features,
        'out_features': linear.out_features,
        'bias': linear.bias is not None,
    }
    return hyper_params
