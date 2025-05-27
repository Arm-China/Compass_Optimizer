# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QSoftmax(QBaseOperator):
    def __init__(self,
                 name,
                 dim=-1,
                 dtype=None) -> None:
        super().__init__(dtype, name=name)
        self.dim = dim
        self.activation_qinfo = QATConfig.get('activation_qinfo')

    @check_args
    def forward(self, inputs, *args):
        self.dim = args[0] if len(args) else self.dim
        outputs = torch.nn.functional.softmax(inputs, self.dim)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def serialize(self, inputs):
        from AIPUBuilder import ops
        out_q = self.get_quantization(self.activation_qinfo)
        return ops.softmax(inputs, axis=self.dim, quantization=out_q)
