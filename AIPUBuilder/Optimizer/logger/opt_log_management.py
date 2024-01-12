# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

import time
import os
import sys

from AIPUBuilder.Optimizer.version import __OPT_VERSION__, __OPT_NAME__
from AIPUBuilder.Optimizer.logger import OPT_DEBUG, OPT_ERROR, OPT_INFO, OPT_WARN, OPT_FATAL

TOOL_NAME = 'Compass-Optimizer' if __OPT_NAME__ is None else __OPT_NAME__
VERSION = '0.0.1' if __OPT_VERSION__ is None else __OPT_VERSION__


class OptLogManagement(object):

    def __init__(self):
        self.log_mode_ = 'console'  # 'file'
        self.log_format_ = 'opt_format'
        self.begin_time_ = time.time()
        self.opt_workflow_footprint_ = ''
        self.opt_workflow_list = []

    def opt_start(self, cfg):
        import torch
        import os
        cuda = torch.cuda.is_available()
        cuda_visible = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        device = f"cuda::{cuda_visible}" if cuda else "cpu"
        OPT_INFO(f"tool name: {TOOL_NAME}, version: {VERSION}, use cuda: {cuda}, running device: {device}")
        cfg_str = '[model name]: ' + cfg.model_name + ', ' + \
                  '[quantization method for weight]: ' + str(cfg.quantize_method_for_weight) + ', ' + \
                  '[quantization method for activation]: ' + str(cfg.quantize_method_for_activation) + ', ' + \
                  '[calibation strategy for weight]: ' + str(cfg.calibration_strategy_for_weight) + ', ' + \
                  '[calibation strategy for activation]: ' + str(cfg.calibration_strategy_for_activation) + ', ' + \
                  '[quantization precision]: activation_bits=%s, weight_bits=%s, bias_bits=%s, lut_items_in_bits=%s' % (
                      str(cfg.activation_bits), str(cfg.weight_bits), str(cfg.bias_bits), str(cfg.lut_items_in_bits))
        OPT_INFO('[quantization config Info]%s\n' % (cfg_str))
        return cfg_str

    def opt_end(self, quantize_summary):
        end_time = time.time()
        cost_time = end_time - self.begin_time_
        OPT_INFO(f"{TOOL_NAME} has done at [{self.opt_workflow_footprint_}] period.")
        summary_keys = {'scale': ['in', 'out'], 'metrics': ['float', 'quant', 'drop']}
        sum_str = ''
        if quantize_summary is not None:
            for k, v in quantize_summary.items():
                sum_str += '[' + k + ']: '
                if isinstance(v, dict):
                    for e in v:
                        sum_str += e + ': ' + str(quantize_summary[k][e]) + ' '
                else:
                    sum_str += str(v)
            # for k, v in summary_keys.items():
            #     if k in quantize_summary.keys():
            #         sum_str += '[' + k + ']: '
            #         for e in v:
            #             if e in quantize_summary[k].keys():
            #                 sum_str += e + ': ' + str(quantize_summary[k][e]) + ' '
            OPT_INFO('[Done]cost time: %ss, and %s' % (str(int(cost_time)), sum_str))
        else:
            OPT_INFO('[Done]cost time: %s' % str(cost_time))


opt_log_manager = OptLogManagement()


def OPT_START(cfg):
    cfg_info = opt_log_manager.opt_start(cfg)
    return cfg_info


def OPT_END(quant_sum=None):
    opt_log_manager.opt_end(quant_sum)


def opt_workflow_register(func):
    def wrapper(*args, **kargs):
        if not (opt_log_manager.opt_workflow_footprint_ in ['metric'] and func.__name__ in ['quant_metric', 'float_metric']):
            opt_log_manager.opt_workflow_footprint_ = func.__name__
            opt_log_manager.opt_workflow_list.append(func.__name__)
            OPT_INFO(f"[{func.__name__}] is running.")
        return func(*args, **kargs)
    return wrapper
