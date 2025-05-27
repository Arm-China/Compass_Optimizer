# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QGeLU(QBaseOperator):
    def __init__(self, name, approximate, dtype=None) -> None:
        super().__init__(dtype)

        self.name = name
        self.approximate = approximate
        self.activation_qinfo = QATConfig.get('activation_qinfo')

    @check_args
    def forward(self, inputs):
        outputs = torch.nn.functional.gelu(input=inputs, approximate=self.approximate)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def serialize(self, input):
        from AIPUBuilder import ops
        out_q = self.get_quantization(self.activation_qinfo)
        gl = ops.gelu(input, quantization=out_q)
        return gl
