# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
import re


def tune_op_passes(graph: PyGraph, config):
    def parse_params_from_tune_op_cfg_line(cfg_str: str):
        s_idx = cfg_str.find(']')
        param_list = re.split(r',|\[|\]|\(|\)|\s+', cfg_str[:s_idx])
        extra_params = [float(param) for param in param_list if param.lower().strip()]
        param_list = re.split(r',|\[|\]|\(|\)|\s+', cfg_str[s_idx:])
        approx_params = [float(param) for param in param_list if param.lower().strip()]
        return extra_params, approx_params

    def parse_multiply_params_from_tune_op_cfg_line(cfg_str: str):
        # such as basiclstm, enable_pass_tune_op_complicated_activations = [0][1,1][0][1,1][0][1,1], corresponding to the three activations
        rint = r'\s*[\-|\+]?\d+\s*'
        rfloat = r'\s*[\-|\+]?\d+(?:\.\d+)?\s*'
        patten = r'(\s*\[{}(?:,{})*\]\s*\[{}(?:,{})*\])'.format(rint, rfloat, rint, rfloat)
        cfg_value = re.findall(patten, cfg_str)
        return cfg_value

    ca_type_list = [OpType.Cosine, OpType.Sine, OpType.Atanh, OpType.Asinh, OpType.Atan, OpType.Asin, OpType.Acos, OpType.Acosh,
                    OpType.Sinh, OpType.Tanh, OpType.Erf, OpType.Log, OpType.SHRINK, OpType.Softsign, OpType.Exp, OpType.Softplus,
                    OpType.Cosh, OpType.MISH, OpType.CELU, OpType.SELU, OpType.ELU, OpType.GELU, OpType.Silu, OpType.Swish, OpType.Sigmoid]
    ca_name_list = [x.name.lower().strip() for x in ca_type_list]

    for n in graph.nodes:
        # tune_op_complicated_activations
        if n.type in ca_type_list or (OpType.Activation == n.type and n.params['method'].lower().strip() in ca_name_list):
            cfg_str = config.enable_pass_tune_op_complicated_activations.get(n)
            n.attrs['extra_params'], n.attrs['approx_params'] = parse_params_from_tune_op_cfg_line(cfg_str)
        # tune_op_softmax
        elif OpType.Softmax == n.type:
            cfg_str = config.enable_pass_tune_op_softmax.get(n)
            n.attrs['extra_params'], n.attrs['approx_params'] = parse_params_from_tune_op_cfg_line(cfg_str)
            method = int(n.attrs['extra_params'][0])
            min_compatible_zhouyi_target = n.attrs["min_compatible_zhouyi_target"].upper()
            if 0 == method:
                sensitive = len(n.children) < 1
                sensitive_optypes = [OpType.NMS, OpType.Sort, OpType.ArgMinMax, OpType.TopK, OpType.InTopK]
                for chl in n.get_descendants()[0]:
                    if chl.type in sensitive_optypes:
                        sensitive = True
                        break
                # use quantize_method=fast_exp if not sensitive or platform lower than x3 else quantize_method=lut
                n.attrs['extra_params'] = [2, 20] if (sensitive or Target.optimized_target_level(
                    min_compatible_zhouyi_target) >= 2) else [1, 1]
        elif OpType.BasicLSTM == n.type:
            cfg_str = config.enable_pass_tune_op_complicated_activations.get(n)
            cfg_value = parse_multiply_params_from_tune_op_cfg_line(cfg_str)
            cfg_value = cfg_value[:3]
            default_extra_params = [[0], [0], [0]]
            default_approx_params = [[1, 1], [1, 1], [1, 1]]
            if len(cfg_value) == 1:
                cfg_value.append(cfg_value[-1])
                cfg_value.append(cfg_value[-1])
            elif len(cfg_value) == 2:
                cfg_value.append(cfg_value[-1])
            else:
                pass
            for idx, params_str in enumerate(cfg_value):
                default_extra_params[idx], default_approx_params[idx] = parse_params_from_tune_op_cfg_line(params_str)
            n.attrs['extra_params'] = default_extra_params
            n.attrs['approx_params'] = default_approx_params
        else:
            pass
