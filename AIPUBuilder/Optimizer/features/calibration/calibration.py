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
    from AIPUBuilder.Optimizer.config import GlobalCalibrationParamField
    valid, methods = GlobalCalibrationParamField._parse(cstrategy)
    if valid:
        OPT_INFO('applying global calibration strategy: ' + str(strategy))
        for method in methods:
            mname = method[0]
            mparams = method[1]
            mscopes = method[2]
            if 'easy_quant' == mname:
                easy_quant_global_calibration(g, cdataloader, mparams, mscopes)
            elif 'adaround' == mname:
                adaround_global_calibration(g, cdataloader, mparams, mscopes)
            elif 'adaquant_zy' == mname:
                adaquant_zy_global_calibration(g, cdataloader, mparams, mscopes)
            elif 'svd_quant' == mname:
                svd_based_quant_global_calibration(g, cdataloader, mparams, mscopes)
            elif 'mvn_correction' == mname:
                from AIPUBuilder.Optimizer.experiment.mvn_correction import mvn_correction_global_calibration
                mvn_correction_global_calibration(g, cdataloader, mparams, mscopes)
            else:
                pass
