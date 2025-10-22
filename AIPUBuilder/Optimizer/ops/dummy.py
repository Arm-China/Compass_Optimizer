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
