# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
import re


def parse_params_from_tune_op_cfg_line(cfg_str: str):
    s_idx = cfg_str.find(']')
    param_list = re.split(r',|\[|\]|\(|\)|\s+', cfg_str[:s_idx])
    extra_params = [float(param) for param in param_list if param.lower().strip()]
    param_list = re.split(r',|\[|\]|\(|\)|\s+', cfg_str[s_idx:])
    approx_params = [float(param) for param in param_list if param.lower().strip()]
    return extra_params, approx_params


def tune_op_complicated_activations(graph: PyGraph, config):
    for n in graph.nodes:
        if OpType.Activation == n.type and n.params['method'].lower() in ['gelu', 'silu', 'swish', 'sigmoid']:
            cfg_str = config.enable_pass_tune_op_complicated_activations.get(n)
            n.attrs['extra_params'], n.attrs['approx_params'] = parse_params_from_tune_op_cfg_line(cfg_str)


def tune_op_softmax(graph: PyGraph, config):
    sensitive_optypes = [OpType.NMS, OpType.Sort, OpType.ArgMinMax, OpType.TopK, OpType.InTopK]
    for n in graph.nodes:
        if OpType.Softmax == n.type:
            cfg_str = config.enable_pass_tune_op_softmax.get(n)
            n.attrs['extra_params'], n.attrs['approx_params'] = parse_params_from_tune_op_cfg_line(cfg_str)
            method = int(n.attrs['extra_params'][0])
            if 0 == method:
                sensitive = len(n.children) < 1
                for chl in n.get_descendants()[0]:
                    if chl.type in sensitive_optypes:
                        sensitive = True
                        break
                # use quantize_method=fast_exp if sensitive else quantize_method=lut
                n.attrs['extra_params'] = [2, 20] if sensitive else [1, 1]


def tune_op_trigonometric_activations(graph: PyGraph, config):
    for n in graph.nodes:
        if n.type in [OpType.Cosine, OpType.Sine, OpType.Atanh, OpType.Asinh, OpType.Atan, OpType.Asin, OpType.Acos, OpType.Acosh,
                      OpType.Sinh, OpType.Tanh, OpType.Erf, OpType.Log, OpType.SHRINK, OpType.Softsign, OpType.Exp, OpType.Softplus,
                      OpType.Cosh, OpType.MISH, OpType.CELU, OpType.SELU, OpType.ELU]:
            cfg_str = config.enable_pass_tune_op_complicated_activations.get(n)
            n.attrs['extra_params'], n.attrs['approx_params'] = parse_params_from_tune_op_cfg_line(cfg_str)
