# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from torch import nn
from .qat_base_operator import QBaseOperator
from ..config import QATConfig


class QActivation(QBaseOperator):
    def __init__(self,
                 name,
                 method,
                 dtype=None,
                 ) -> None:
        super().__init__(dtype)
        self.name = name
        self.method = method

    def forward(self, inputs):
        pass

    def serialize(self, input):
        pass
