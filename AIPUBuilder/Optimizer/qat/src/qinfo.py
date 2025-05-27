# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import os
import torch
import numpy as np
from enum import Enum
from collections import namedtuple, OrderedDict
from AIPUBuilder.Optimizer.framework import Dtype
from AIPUBuilder.Optimizer.utils import torch_tensor
from .qatlogger import QAT_ERROR


class QScheme(Enum):
    per_tensor_symmetric_restricted_range = 'per_tensor_symmetric_restricted_range'
    per_tensor_symmetric_full_range = 'per_tensor_symmetric_full_range'
    per_tensor_asymmetric = 'per_tensor_asymmetric'
    per_channel_symmetric_restricted_range = 'per_channel_symmetric_restricted_range'
    per_channel_symmetric_full_range = 'per_channel_symmetric_full_range'
    per_channel_asymmetric_full_range = 'per_channel_asymmetric'

    @staticmethod
    def str_to_qscheme(name):
        if isinstance(name, QScheme):
            return name
        if not hasattr(QScheme, name):
            QAT_ERROR(f"unsupport QScheme {name}, which now only support {QScheme.__members__}")
        return QScheme.__members__[name]

    @staticmethod
    def str(name):
        membs = dict(zip(QScheme.__members__.values(), QScheme.__members__.keys()))
        if name in membs:
            return membs[name]
        else:
            QAT_ERROR(f"unsupport QScheme {name}, which now only support {QScheme.__members__}")

    @staticmethod
    def is_perchannel(qscheme):
        if isinstance(qscheme, QScheme):
            str_qscheme = QScheme.str(qscheme)
            return True if 'channel' in str_qscheme else False
        elif isinstance(qscheme, str):
            return True if 'channel' in str_qscheme else False
        else:
            QAT_ERROR(f"unsupport type of args for is_perchannel")
            return False

    @staticmethod
    def to_symmetric(qscheme):
        str_qscheme = qscheme
        if isinstance(qscheme, QScheme):
            str_qscheme = QScheme.str(qscheme)
        if 'asymmetric' in str_qscheme:
            str_qscheme = str_qscheme[:str_qscheme.index('asymmetric')] + 'symmetric_full_range'
        return QScheme.str_to_qscheme(str_qscheme)


class CMode(Enum):
    extrema = 'extrema'
    mean = 'mean'

    @staticmethod
    def str_to_cmode(name):
        if isinstance(name, CMode):
            return name
        if not hasattr(CMode, name):
            QAT_ERROR(f"unsupport CMode {name}, which now only support {CMode.__members__}")
        return CMode.__members__[name]

    @staticmethod
    def str(name):
        membs = dict(zip(CMode.__members__.values(), CMode.__members__.keys()))
        if name in membs:
            return membs[name]
        else:
            QAT_ERROR(f"unsupport CMode {name}, which now only support {CMode.__members__}")


class QuantStage(Enum):
    FP32 = 'fp32'
    CALIB = 'calib'
    QAT = 'qat'
    INFER = 'infer'  # full quant inference
    INFER_SHAPE = 'infer_shape'

    @staticmethod
    def str_to_quantstage(stage):
        if isinstance(stage, QuantStage):
            return stage
        if not isinstance(stage, str):
            QAT_ERROR(
                f"the arg of QuantStage.str_toquantstage which should be str or QuantStage, but now is {type(stage)}")
            return None

        stage = stage.upper()
        if not hasattr(QuantStage, stage):
            QAT_ERROR(f"unsupport QuantStage {stage}, which now only support {QuantStage.__members__}")
        return QuantStage.__members__[stage]


class QInfos(namedtuple("QInfos", ["activation", "weight", "bias"])):
    def __new__(cls, activation, weight, bias):
        return super(QInfos, cls).__new__(cls, activation, weight, bias)


class QInfo(object):
    def __init__(self, qconfig=None):
        self._scale = 1.0
        self._zerop = 0
        self._dtype = None

        self._maxs = None
        self._mins = None
        self._qmax = None
        self._qmin = None

        self._bits = 8
        self._bias_effective_bits = ''
        self._qscheme = 'per_tensor_symmetric_full_range'  # quantize mode
        self._cmode = 'extrema'  # calibrate mode
        self._key_axis = 0
        self._running_momentum = 0.9
        self._group = 1

        self._qinvariant = False
        self._unquantifiable = False

        self._device = 'cpu'

    @property
    def is_quantized(self):
        if self.mins is None and self.maxs is None and self._scale == 1.0 and self._zerop == 0:
            return False
        return True

    @property
    def cmode(self):
        return self._cmode

    @cmode.setter
    def cmode(self, cm):
        if isinstance(cm, str):
            self._cmode = CMode.str_to_cmode(cm)
        elif isinstance(cm, CMode):
            self._cmode = cm
        else:
            QAT_ERROR(f"unsupported cmode type(={type(cm)}), now only support: [str, CMode]")

    @property
    def qinvariant(self):
        return self._qinvariant

    @qinvariant.setter
    def qinvariant(self, ka):
        self._qinvariant = ka

    @property
    def key_axis(self):
        return self._key_axis

    @key_axis.setter
    def key_axis(self, ka):
        self._key_axis = ka

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, group):
        self._group = group

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @property
    def qscheme(self):
        return self._qscheme if isinstance(self._qscheme, QScheme) else QScheme.str_to_qscheme(self._qscheme)

    @qscheme.setter
    def qscheme(self, quantize_mode):
        if isinstance(quantize_mode, str):
            self._qscheme = QScheme.str_to_qscheme(quantize_mode)
        elif isinstance(quantize_mode, QScheme):
            self._qscheme = quantize_mode
        else:
            QAT_ERROR(
                f"unsupported type(={type(quantize_mode)}) when setting qscheme, now only supported: [str, QScheme]")

    @property
    def qmin(self):
        return self._qmin

    @qmin.setter
    def qmin(self, q_min):
        self._qmin = q_min

    @property
    def qmax(self):
        return self._qmax

    @qmax.setter
    def qmax(self, q_max):
        self._qmax = q_max

    @property
    def bits(self):
        return self._bits

    @bits.setter
    def bits(self, quantize_bits):
        if not quantize_bits in range(3, 65):
            QAT_ERROR(f"set bits is out of range [3, 64]")
            quantize_bits = min(max(quantize_bits, 3), 64)
        self._bits = quantize_bits

    @property
    def bias_effective_bits(self):
        return self._bias_effective_bits

    @bias_effective_bits.setter
    def bias_effective_bits(self, quantize_bits):
        if isinstance(quantize_bits, str) and quantize_bits == '':
            self._bias_effective_bits = ''
            return
        if not quantize_bits in range(16, 49):
            QAT_ERROR(f"set bits is out of range [16,48]")
            quantize_bits = min(max(quantize_bits, 16), 48)
        self._bias_effective_bits = quantize_bits

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, data_type):
        self._dtype = data_type

    @property
    def scale(self):
        return torch_tensor(self._scale, device=self.device).reshape([-1]).float()

    @scale.setter
    def scale(self, scale):
        if isinstance(scale, torch.Tensor):
            self._scale = scale.reshape([-1]).float().to(self.device)
        elif scale is None:
            QAT_ERROR(f"set scale is None, please check the setted scale")
        else:
            self.scale = torch_tensor(scale, device=self.device)

    @property
    def zerop(self):
        return torch_tensor(self._zerop, device=self.device).reshape([-1]).int()

    @zerop.setter
    def zerop(self, zerop):
        if isinstance(zerop, torch.Tensor):
            self._zerop = zerop.reshape([-1]).int().to(self.device)
        elif zerop is None:
            QAT_ERROR(f"set zerop is None, please check the setted zerop")
        else:
            self.zerop = torch_tensor(zerop, device=self.device)

    @property
    def maxs(self):
        if self._maxs is not None:
            self._maxs = torch_tensor(self._maxs, device=self.device).reshape([-1])
        return self._maxs  # pylint: disable=no-member

    @maxs.setter
    def maxs(self, max_v):
        if isinstance(max_v, torch.Tensor):
            self._maxs = max_v.reshape([-1])
        elif max_v is None:
            self._maxs = None
        else:
            self.maxs = torch_tensor(max_v, device=self.device)

    @property
    def mins(self):
        if self._mins is not None:
            self._mins = torch_tensor(self._mins, device=self.device).reshape([-1])
        return self._mins  # pylint: disable=no-member

    @mins.setter
    def mins(self, min_v):
        if isinstance(min_v, torch.Tensor):
            self._mins = min_v.reshape([-1])
        elif min_v is None:
            self._mins = None
        else:
            self.mins = torch_tensor(min_v, device=self.device)

    @property
    def unquantifiable(self):
        return self._unquantifiable

    @unquantifiable.setter
    def unquantifiable(self, uq):
        self._unquantifiable = uq

    @property
    def running_momentum(self):
        return self._running_momentum

    @running_momentum.setter
    def running_momentum(self, uq):
        self._running_momentum = uq

    @staticmethod
    def broadcast(data, shape_len, key_axis=0):

        if isinstance(data, torch.Tensor):
            if data.numel() > 1:
                bc_shape = [1] * shape_len
                bc_shape[key_axis] = -1
            else:
                bc_shape = [1]
            return data.reshape(bc_shape)
        return data

    @classmethod
    def from_qconfig(cls, qconfig=None):
        return cls(qconfig)

    def get_qparams(self, to_torch_quantization_format=True):
        _aipu_to_torch_dtype = {
            Dtype.INT8: torch.qint8,
            Dtype.UINT8: torch.quint8,
            Dtype.INT32: torch.qint32,
        }

        scale = self.scale
        zerop = self.zerop
        dtype = self.dtype
        key_axis = self.key_axis
        if to_torch_quantization_format:
            scale = 1.0 / scale
            zerop = 0 - zerop
        if dtype not in _aipu_to_torch_dtype:
            QAT_ERROR(
                f"when getting qparams, the {dtype} not in _aipu_to_torch_dtype, which only supported{_aipu_to_torch_dtype.values()}")
            return {}
        dtype = _aipu_to_torch_dtype[dtype]
        if scale.numel() == 1:
            qparams = OrderedDict({"_scale_": scale.float().item(),
                                  "_zero_point_": zerop.int().item(), "_dtype_": dtype})
        else:
            qparams = OrderedDict({"_scale_": scale, "_zero_point_": zerop, "_axis_": key_axis, "_dtype_": dtype})
        return qparams

    def clone(self, other):
        if isinstance(other, QInfo):
            for k, v in other.__dict__.items():
                self.__setattr__(k, v)
        return self

    def __repr__(self):
        s = ''
        s += f"scale={str(self.scale.cpu().numpy().tolist())}, "
        s += f"zp={str(self.zerop.cpu().numpy().tolist())}, "
        s += f"dtype={self.dtype}, "
        if self.mins is not None:
            s += f"fmin={str(self.mins.cpu().numpy().tolist())}, "
        if self.maxs is not None:
            s += f"fmax={str(self.maxs.cpu().numpy().tolist())}, "
        s += f"calibration_strategy={self.cmode}, "
        return s
