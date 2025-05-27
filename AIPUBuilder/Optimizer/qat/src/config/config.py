# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import os
# from easydict import EasyDict
import torch
from enum import Enum
from collections import OrderedDict, namedtuple

# from torch.ao.quantization import QConfig as TorchQConfig
from AIPUBuilder.Optimizer.framework import Dtype
from AIPUBuilder.Optimizer.config import CfgParser
from AIPUBuilder.Optimizer.framework import opt_use_cuda
from ..qatlogger import QAT_ERROR
from ..qinfo import QInfo, QScheme, CMode


def default_device():
    return torch.device('cuda:0') if opt_use_cuda else torch.device('cpu')


def get_device():
    if opt_use_cuda():
        # device = os.getenv('CUDA_VISIBLE_DEVICES', '0').split(',')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


class QATConfig:
    qat_config = {}

    def __init__(self, config):
        # self._config = config
        QATConfig.update(config)
        self.set('weight_qinfo', self._get_weight_qinfo)
        self.set('activation_qinfo', self._get_activation_qinfo)
        self.set('bias_qinfo', self._get_bias_qinfo)
        self.set('lut_qinfo', self._get_lut_qinfo)
        self.set('device', get_device())

    @staticmethod
    def update(config):
        for var_name, var_value in config.__dict__.items():
            QATConfig.qat_config.update({var_name: var_value})

    def customize_field(self, field):
        return field() if callable(field) else field

    def _get_common_qinfo(self, qinfo):
        qinfo.device = get_device()
        qinfo.running_mometum = self.get("running_statistic_momentum")

    def _get_weight_qinfo(self):
        self.weight_qinfo = QInfo()
        self._get_common_qinfo(self.weight_qinfo)

        self.weight_qinfo.bits = self.get('weight_bits', 8)
        self.weight_qinfo.qscheme = QScheme.str_to_qscheme(self.get(
            'quantize_method_for_weight', QScheme.per_tensor_symmetric_restricted_range))
        self.weight_qinfo.cmode = CMode.str_to_cmode(self.get(
            'calibration_strategy_for_weight', CMode.extrema))
        return self.weight_qinfo

    def _get_bias_qinfo(self):
        self.bias_qinfo = QInfo()
        self._get_common_qinfo(self.bias_qinfo)

        self.bias_qinfo.bits = self.get('bias_bits')
        self.bias_qinfo.qscheme = QScheme.str_to_qscheme(self.get(
            'quantize_method_for_weight', QScheme.per_tensor_symmetric_restricted_range))
        self.bias_qinfo.cmode = CMode.str_to_cmode(self.get(
            'calibration_strategy_for_weight', CMode.extrema))
        self.bias_qinfo.bias_effective_bits = self.get("bias_effective_bits", '')
        return self.bias_qinfo

    def _get_activation_qinfo(self):
        self.activation_qinfo = QInfo()
        self._get_common_qinfo(self.activation_qinfo)

        # import pdb;pdb.set_trace()
        self.activation_qinfo.bits = self.get('activation_bits')
        self.activation_qinfo.qscheme = QScheme.str_to_qscheme(self.get(
            'quantize_method_for_activation', QScheme.per_tensor_symmetric_full_range))
        self.activation_qinfo.cmode = CMode.str_to_cmode(self.get(
            'calibration_strategy_for_activation', CMode.mean))
        self.activation_qinfo.running_momentum = self.get('running_statistic_momentum', 0.95)

        return self.activation_qinfo

    def _get_lut_qinfo(self):
        self.lut_qinfo = QInfo()
        self._get_common_qinfo(self.lut_qinfo)

        self.lut_qinfo.bits = self.get('lut_items_in_bits')
        self.lut_qinfo.qscheme = QScheme.str_to_qscheme(self.get(
            'quantize_method_for_activation', QScheme.per_tensor_symmetric_full_range))
        self.lut_qinfo.cmode = CMode.str_to_cmode(self.get(
            'calibration_strategy_for_activation', CMode.extrema))
        return self.lut_qinfo

    @staticmethod
    def get(name, default=None):

        if name in QATConfig.qat_config:
            field = QATConfig.qat_config[name]
            return field() if callable(field) else field
        else:
            if default is None:
                QAT_ERROR(f"{name} not in qat_config, and not set default value")
        return default

    @staticmethod
    def set(key, value):
        QATConfig.qat_config.update({key: value})
