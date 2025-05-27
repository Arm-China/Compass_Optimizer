# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


import torch
from torch import nn
from torch.nn.parameter import Parameter
from ..qatlogger import QAT_ERROR
from ..qinfo import QuantStage
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator


@register_operator()
class QConstant(QBaseOperator):
    def __init__(self, name, data, dtype=None) -> None:
        super().__init__(dtype)
        self.name = name
        if data is None:
            QAT_ERROR(f"when instances one QConstant, the data arg is None")
        self.register_buffer('weight', data)
        self.activation_qinfo = QATConfig.get('activation_qinfo')

    def forward(self):
        if not torch.is_floating_point(self.weight):
            self.activation_qinfo.qinvariant = True
        outputs = self.fake_quant(self.weight, self.activation_qinfo)
        return outputs

    def serialize(self):
        from AIPUBuilder import ops
        from AIPUBuilder.core import Tensor
        weight = Tensor(self.name + "_weight", self.weight.cpu().numpy().astype('float32'))
        if self.ir_mode == 'fp':
            out = ops.constant(weight)
        else:
            bk = self.quant_stage
            self.quant_stage = QuantStage.QAT
            self.forward()
            self.quant_stage = bk
            out_q = self.get_quantization(self.activation_qinfo)
            out = ops.constant(weight, quantization=out_q)
        return out
