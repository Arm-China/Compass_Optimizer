# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QDropOut(QBaseOperator):
    def __init__(self,
                 p=0.0,
                 dtype=None,
                 name='') -> None:
        super().__init__(dtype)
        self._use_input_QConfig = True
        self.p = p
        self.activation_qinfo = QATConfig.get('activation_qinfo')
        self.name = name

    @check_args
    def forward(self, inputs, *args):
        outputs = self.fake_quant(inputs, self.activation_qinfo)
        return outputs

    def serialize(self, input):
        return input
