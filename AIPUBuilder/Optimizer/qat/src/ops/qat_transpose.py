# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from ..qatlogger import QAT_ERROR
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QTranspose(QBaseOperator):
    def __init__(self,
                 name,
                 perm,
                 dtype=None) -> None:
        super().__init__(dtype, name=name)
        self._use_input_QConfig = True
        self.perm = perm
        self.activation_qinfo = QATConfig.get('activation_qinfo')

    def forward(self, inputs, perm=None):
        if len(self.perm) == 0 and perm is None:
            QAT_ERROR(f"QTranspose meets the len(perm) == 0.")
        self.perm = perm if perm is not None else self.perm
        outputs = torch.permute(inputs, self.perm)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def serialize(self, inputs):
        from AIPUBuilder import ops
        return ops.transpose(inputs, list(self.perm))
