# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import torch

register_optype('Switch')


@op_register(OpType.Switch)
def switch__forward(self, *args):
    target = self.inputs[0].betensor
    for inp in self.inputs:
        if inp.betensor.numel() > 0:
            target = inp.betensor
            break
    self.outputs[0].betensor = target
    return self.outputs[0].betensor


@quant_register(OpType.Switch)
def switch__quantize(self, *args):
    q_bits_activation = self.attrs["q_bits_activation"]
    q_mode_activation = self.attrs["q_mode_activation"]
    for ot, it in zip(self.outputs, self.inputs):
        if it.qbits is not None:
            ot.clone_qinfo(it)
        else:
            if is_float(it.ir_dtype):
                ot.qinvariant = False
                ot.qbits = q_bits_activation
                ot.scale, ot.zerop, ot.qmin, ot.qmax, ot.dtype = get_linear_quant_params_from_tensor(
                    ot, q_mode_activation, ot.qbits, is_signed=True)
            else:
                ot.scale = 1.0
                ot.zerop = 0
                ot.qbits = dtype2bits(it.ir_dtype)
                ot.dtype = it.ir_dtype
                ot.qinvariant = True
