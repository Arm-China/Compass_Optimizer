# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import torch

register_optype('Dummy')


@op_register(OpType.Dummy)
def dummy__forward(self, *args):
    for ot, it in zip(self.outputs, self.inputs):
        ot.betensor = it.betensor.clone()


@quant_register(OpType.Dummy)
def dummy__quantize(self, *args):
    for ot, it in zip(self.outputs, self.inputs):
        ot.clone_qinfo(it)
