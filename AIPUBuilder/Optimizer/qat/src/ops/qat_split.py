# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QSplit(QBaseOperator):
    def __init__(self,
                 name,
                 split_size_or_sections,
                 dim=0,
                 dtype=None) -> None:
        super().__init__(dtype, name=name)
        self._use_input_QConfig = True
        self.dim = dim
        self.split_size_or_sections = split_size_or_sections
        self.activation_qinfo = QATConfig.get('activation_qinfo')

    @check_args
    def forward(self, inputs, *args):
        self.split_size_or_sections = args[0] if len(args) else self.split_size_or_sections
        self.dim = args[1] if len(args) > 1 else self.dim
        outputs = torch.split(inputs, self.split_size_or_sections, self.dim)
        outputs = list(outputs)
        for i, out in enumerate(outputs):
            outputs[i] = self.fake_quant(out, self.activation_qinfo)
        return outputs

    def serialize(self, inputs):
        from AIPUBuilder import ops
        ops_split_size_or_sections = self.split_size_or_sections
        if isinstance(self.split_size_or_sections, int):
            s = inputs.shape[self.dim]
            ops_split_size_or_sections = s // self.split_size_or_sections

        return ops.split(inputs, splits=ops_split_size_or_sections, axis=self.dim)
