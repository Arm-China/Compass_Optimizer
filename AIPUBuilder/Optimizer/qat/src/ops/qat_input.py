# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder import ops
from .qat_base_operator import QBaseOperator, check_args
from ..qatregister import register_operator
from ..qatlogger import QAT_ERROR
from ..config import QATConfig
import numpy as np
import torch


@register_operator()
class QInput(QBaseOperator):
    def __init__(self, shape=None, name=None, dtype=None):
        super().__init__(dtype)
        self.activation_qinfo = QATConfig.get('activation_qinfo')
        self._input_shape = shape
        self._original_dtype = None
        self.name = name

    @check_args
    def forward(self, input_data):
        self._input_shape = input_data.shape
        if not torch.is_floating_point(input_data):
            self.activation_qinfo.qinvariant = True
            self._original_dtype = input_data.dtype
        return self.fake_quant(input_data, self.activation_qinfo)

    def _get_shape_dtype(self, shape, dtype):
        if shape is not None:
            self._input_shape = shape
        if dtype is not None:
            self._original_dtype = dtype

    def _get_dtype(self):
        from AIPUBuilder.core import Dtype as cDtype
        from AIPUBuilder.Optimizer.framework import Dtype as pDtype
        dtype = cDtype.FP32
        if self._original_dtype is not None:
            if isinstance(self._original_dtype, torch.dtype):
                dtype = cDtype.from_torch(self._original_dtype)
            elif isinstance(self._original_dtype, cDtype):
                dtype = self._original_dtype
            elif isinstance(self._original_dtype, pDtype):
                dtype = cDtype(self._original_dtype.name.lower())
            elif isinstance(self._original_dtype, np.dtype):
                dtype = cDtype.from_np(self._original_dtype)
            else:
                QAT_ERROR(f"unsupported dtype = ({type(self._original_dtype)})")
        return dtype

    def serialize(self, input_or_shape=None, dtype=None):
        from AIPUBuilder.core import Tensor, Dtype, TensorShape

        shape = input.shape if isinstance(input_or_shape, (torch.Tensor, np.ndarray, Tensor)) else input_or_shape
        self._get_shape_dtype(shape, dtype)

        if self.ir_mode == 'fp':
            dtype = self._get_dtype()
        else:
            if self.activation_qinfo.dtype is not None:
                dtype = Dtype(self.activation_qinfo.dtype.name.lower())
            else:
                dtype = self._get_dtype()

        if isinstance(input, torch.Tensor):
            inpt = Tensor(input.cpu().numpy(), dtype.np)
        elif isinstance(input, np.ndarray):
            inpt = Tensor(input, dtype.np)
        elif isinstance(input, Tensor):
            inpt = input
        elif isinstance(input, (list, tuple)) and (self._input_shape is not None and len(self._input_shape) > 0 and len(input) > 0):
            inpt = Tensor(TensorShape(input), dtype.np)
        else:
            inpt = Tensor(TensorShape(list(self._input_shape)), dtype)

        if self.ir_mode == 'fp':
            inpt.quantization.using_qtlib = False
        else:
            inpt.quantization.using_qtlib = False
            self.update_tensor_quantization(inpt, self.activation_qinfo)
        return inpt
