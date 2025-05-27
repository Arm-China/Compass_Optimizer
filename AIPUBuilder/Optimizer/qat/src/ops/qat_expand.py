# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from ..qatlogger import QAT_ERROR
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QExpand(QBaseOperator):
    def __init__(self, size, dtype=None) -> None:
        super().__init__(dtype)
        self._use_input_QConfig = True
        self.size = size
        self.activation_qinfo = QATConfig.get('activation_qinfo')

    @check_args
    def forward(self, input, *args):
        if len(self.size) == 0 and len(args) == 0:
            QAT_ERROR(f"QExpand meets the len(size) == 0.")
        self.size = list(args) if len(args) else self.size
        outputs = input.expand(self.size)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def serialize(self, inputs):
        from AIPUBuilder import ops
        repeats = [s if s != -1 else 1 for s in self.size]
        repeats = [s if s != inputs.shape[i] else 1 for i, s in enumerate(repeats)]
        out = ops.tile(inputs, repeats)
        out.quantization = inputs.quantization
        return out
