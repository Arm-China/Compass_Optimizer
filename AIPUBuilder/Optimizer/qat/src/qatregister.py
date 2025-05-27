# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import abc
from collections import OrderedDict
from .qatlogger import QAT_WARN

QAT_FUSION_PATTERNS = OrderedDict()
QAT_COMPASS_OPERATORS = OrderedDict()


def register_fusion_pattern(pattern):
    def insert(fn):
        if pattern in QAT_FUSION_PATTERNS.keys():
            QAT_WARN(f"QAT Pattern {pattern} has already registered, and will be overwritten")
        QAT_FUSION_PATTERNS[pattern] = fn
        return fn
    return insert


def register_operator(optype=None):
    def insert(fn):
        if fn in QAT_COMPASS_OPERATORS.keys():
            QAT_WARN(f"QAT compass operater({fn}) has already registered, and will be overwritten")
        if optype is None:
            QAT_COMPASS_OPERATORS[fn] = True
        else:
            QAT_COMPASS_OPERATORS[fn] = optype
        return fn
    return insert


def get_default_fusion_patterns():
    return QAT_FUSION_PATTERNS


def get_compass_supported_operators():
    '''tuple of supported operators!'''
    return tuple(QAT_COMPASS_OPERATORS.keys())


class QATBaseTrainLoop(object):
    # def __init__(self, *args, **kwargs):
    #     pass
    def set_stage(self, model, stage='qat'):
        from .qinfo import QuantStage
        for m in model.modules():
            if isinstance(m, tuple(QAT_COMPASS_OPERATORS.keys())):
                m.quant_stage = QuantStage.str_to_quantstage(stage)
    # @abc

    def __call__(self, model, *args, **kwargs):
        pass
