# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from ..qatregister import register_operator
from ..qinfo import CMode
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QHardSigmoid(QBaseOperator):
    def __init__(self, name, dtype=None) -> None:
        super().__init__(dtype)

        self.name = name
        self.method = "HARDSIGMOID"
        self.clip_min = 0.0
        self.clip_max = 1.0
        self.activation_qinfo = QATConfig.get('activation_qinfo')
        self.activation_qinfo.cmode = 'extrema'

    def forward(self, inputs):
        outputs = torch.nn.functional.hardsigmoid(inputs)
        dev = outputs.device
        outputs = torch.maximum(torch.tensor(self.clip_min, device=dev),
                                torch.minimum(outputs, torch.tensor(self.clip_max, device=dev)))
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def serialize(self, input):
        from AIPUBuilder import ops
        out_q = self.get_quantization(self.activation_qinfo)
        hs = ops.hard_sigmoid(input, quantization=out_q)
        hs.op.name = self.name
        return hs
