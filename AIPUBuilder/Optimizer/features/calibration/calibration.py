# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.logger import *
from . local_calibration import *
from . global_calibration import *
import torch
import re


def apply_calibration_strategy(t, strategy, quantize_method):
    if strategy:  # None means pass
        cstrategy = strategy.lower().strip()
        if cstrategy == 'extrema':
            extrema_calibration(t)
        elif cstrategy == 'in_ir':
            in_ir_calibration(t)
        elif cstrategy == 'mean':
            mean_calibration(t)
        elif re.match(r'^weighted_scale_param.*$', cstrategy):
            weighted_scale_param_calibration(t, cstrategy)
        elif re.match(r'^\d+std$', cstrategy):  # eg. 3std/5std/10std...
            nstd_calibration(t, cstrategy)
        elif re.match(r'^\d*kld$', cstrategy):  # eg. 5kld/10kld/20kld
            nkld_calibration(t, cstrategy)
        elif re.match(r'^(\d\.?\d*)*aciq_laplace$', cstrategy):
            aciq_laplace_calibration(t, cstrategy, quantize_method)
        elif re.match(r'^(\d\.?\d*)*aciq_gauss$', cstrategy):
            aciq_gauss_calibration(t, cstrategy, quantize_method)
        elif re.match(r'^(\d\.?\d*)*percentile$', cstrategy):
            percentile_calibration(t, cstrategy)
        else:
            OPT_WARN("unsupported calibration strategy: %s" % strategy)
            t.min = t.running_min
            t.max = t.running_max
            if None != t.running_min_key_axis:
                t.min_key_axis = t.running_min_key_axis
                t.max_key_axis = t.running_max_key_axis
    t.min = min(t.min, 0.0)
    t.max = max(t.max, 0.0)
    if None != t.min_key_axis:
        t.min_key_axis = torch.min(t.min_key_axis, torch.zeros_like(t.min_key_axis))
        t.max_key_axis = torch.max(t.max_key_axis, torch.zeros_like(t.max_key_axis))


def apply_global_calibration(g, cdataloader, strategy):
    cstrategy = strategy.lower().strip()
    if cstrategy != 'none':
        OPT_INFO('applying global calibration strategy: ' + str(strategy))
        if len(re.findall('easy_quant', cstrategy)) > 0:
            easy_quant_global_calibration(g, cdataloader, cstrategy)
        elif len(re.findall('adaround', cstrategy)) > 0:
            adaround_global_calibration(g, cdataloader, cstrategy)
        elif len(re.findall('svd_quant', cstrategy)) > 0:
            from AIPUBuilder.Optimizer.experiment.svd_based_calibration import svd_based_quant_global_calibration
            svd_based_quant_global_calibration(g, cdataloader, cstrategy)
        else:
            pass
