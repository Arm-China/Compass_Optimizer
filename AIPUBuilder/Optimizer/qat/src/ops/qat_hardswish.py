# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from torch import nn
from AIPUBuilder.Optimizer.framework import OpType
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QHardSwish(QBaseOperator):
    def __init__(self, name, dtype=None) -> None:
        super().__init__(dtype)

        self.name = name
        self.method = "HARDSWISH"
        self.activation_qinfo = QATConfig.get('activation_qinfo')
        self.activation_qinfo.cmode = 'extrema'

    def forward(self, inputs):
        outputs = torch.nn.functional.hardswish(inputs)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def serialize(self, input):
        from AIPUBuilder import ops
        out_q = self.get_quantization(self.activation_qinfo)
        hs = ops.hard_swish(input, quantization=out_q)
        hs.op.name = self.name
        return hs
