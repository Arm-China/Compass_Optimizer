# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QMatMul(QBaseOperator):
    def __init__(self,
                 name,
                 trans_a=False,
                 trans_b=False,
                 dtype=None) -> None:
        super().__init__(dtype, name=name)
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.activation_qinfo = QATConfig.get('activation_qinfo')

    @check_args
    def forward(self, x, y):
        if self.trans_a:
            if x.dim() == 0:
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 1:
                x = x.unsqueeze(0)
            x = x.transpose(-1, -2)
        if self.trans_b:
            if y.dim() == 0:
                y = y.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 1:
                y = y.unsqueeze(0)
            y = y.transpose(-1, -2)
        outputs = torch.matmul(x, y)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def serialize(self, input0, input1):
        from AIPUBuilder import ops
        out_q = self.get_quantization(self.activation_qinfo)
        return ops.matmul(input0, input1, self.trans_a, self.trans_b, quantization=out_q)
