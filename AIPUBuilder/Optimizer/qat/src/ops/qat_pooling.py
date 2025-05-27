# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from torch import nn
from ..utils import (
    convert_adaptive2avg_pool_hyperparams,
    extract_avgpool_hyperparams,
    convert_adaptive2max_pool_hyperparams,
    extract_maxpool_hyperparams,
)

from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


class QPooling(QBaseOperator):
    def __init__(self, dtype=None, name='') -> None:
        super().__init__(dtype)
        self._use_input_QConfig = True
        self.activation_qinfo = QATConfig.get('activation_qinfo')
        self.method = 'NONE'
        self._is_module_params_set = False
        self._input_shape = None
        self.name = name

    # def serialize(self, input):
    #     from AIPUBuilder import ops
    #     from AIPUBuilder.core import Dtype
    #     # TODO
    #     self.dilation = (1, 1)
    #
    #     input_trans = ops.transpose(input, perm=[0, 2, 3, 1])
    #     if self.method == "MAX":
    #         pool_out = ops.max_pool(input_trans,
    #                                 self.kernel_size,
    #                                 self.stride,
    #                                 self.dilation,
    #                                 [self.padding[0]]*4,
    #                                 self.ceil_mode)
    #     else:
    #         if self._use_input_QConfig:
    #             out_q = input.quantization
    #         else:
    #             out_q = self.get_quantization(self.activation_qinfo)
    #         pool_out = ops.avg_pool(input_trans,
    #                                 self.kernel_size,
    #                                 self.stride,
    #                                 self.dilation,
    #                                 (self.padding[0], self.padding[1], self.padding[0], self.padding[1]),
    #                                 self.ceil_mode,
    #                                 self.count_include_pad,
    #                                 quantization=out_q)
    #     pool_trans_out = ops.transpose(pool_out, perm=[0, 3, 1, 2])
    #     return pool_trans_out


@register_operator()
class QAveragePooling2D(QPooling):
    def __init__(self,
                 avg_pool,
                 name,
                 dtype=None,
                 ) -> None:
        super().__init__(dtype, name)
        self.avg_pool = avg_pool
        self.method = 'AVG'

    def forward(self, inputs):
        self.set_module_params(inputs)
        outputs = self.avg_pool(inputs)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def set_module_params(self, inputs):
        self._input_shape = list(inputs.shape)
        if isinstance(self.avg_pool, nn.AdaptiveAvgPool2d):
            hyper_params = convert_adaptive2avg_pool_hyperparams(self.avg_pool, self._input_shape)
        else:
            hyper_params = extract_avgpool_hyperparams(self.avg_pool)

        for key in hyper_params:
            setattr(self, key, hyper_params[key])

        self._is_module_params_set = True

    def serialize(self, input):
        from AIPUBuilder import ops
        from AIPUBuilder.core import Dtype
        # TODO
        self.set_module_params(input)
        self.dilation = (1, 1)

        input_trans = ops.transpose(input, perm=[0, 2, 3, 1])
        if self._use_input_QConfig:
            out_q = input.quantization
        else:
            out_q = self.get_quantization(self.activation_qinfo)
        pool_out = ops.avg_pool(input_trans,
                                self.kernel_size,
                                self.stride,
                                self.dilation,
                                (self.padding[0], self.padding[1], self.padding[0], self.padding[1]),
                                self.ceil_mode,
                                self.count_include_pad,
                                quantization=out_q)
        pool_trans_out = ops.transpose(pool_out, perm=[0, 3, 1, 2])
        return pool_trans_out


@register_operator()
class QMaxPooling2D(QPooling):
    def __init__(self,
                 max_pool,
                 name,
                 dtype=None,
                 ) -> None:
        super().__init__(dtype, name)
        self.max_pool = max_pool
        self.method = 'MAX'

    def forward(self, inputs):
        self.set_module_params(inputs)
        outputs = self.max_pool(inputs)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def set_module_params(self, inputs):
        self._input_shape = list(inputs.shape)
        if isinstance(self.max_pool, nn.AdaptiveMaxPool2d):
            hyper_params = convert_adaptive2max_pool_hyperparams(self.max_pool, self._input_shape)
        else:
            hyper_params = extract_maxpool_hyperparams(self.max_pool)

        for key in hyper_params:
            setattr(self, key, hyper_params[key])

        self._is_module_params_set = True

    def serialize(self, input):
        from AIPUBuilder import ops
        # TODO
        self.set_module_params(input)
        self.dilation = (1, 1)
        input_trans = ops.transpose(input, perm=[0, 2, 3, 1])
        pool_out = ops.max_pool(input_trans,
                                self.kernel_size,
                                self.stride,
                                self.dilation,
                                [self.padding[0]]*4,
                                self.ceil_mode)
        pool_trans_out = ops.transpose(pool_out, perm=[0, 3, 1, 2])
        return pool_trans_out
