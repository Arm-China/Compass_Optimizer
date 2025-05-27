# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from torch import nn
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QConcat(QBaseOperator):
    def __init__(self, dim=0, dtype=None, name='') -> None:
        super().__init__(dtype, name)
        self.dim = dim
        self.activation_qinfo = QATConfig.get('activation_qinfo')

    @check_args
    def forward(self, inputs, *args):
        self.dim = args[0] if len(args) else self.dim
        outputs = torch.cat(inputs, dim=self.dim)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def serialize(self, inputs):
        from AIPUBuilder import ops
        out_q = self.get_quantization(self.activation_qinfo)
        out = ops.concat(inputs, axis=self.dim, quantization=out_q)
        return out
