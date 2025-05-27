# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.groupnorm import groupnorm_quantize, groupnorm
import torch


@quant_register(OpType.LayerNorm)
def layernorm_quantize(self, *args):
    groupnorm_quantize(self, *args)


@op_register(OpType.LayerNorm)
def layernorm(self, *args):
    aflag = False
    if 'axis' not in self.params:
        aflag = True
        self.params['axis'] = [-1]
    gflag = False
    if 'group' not in self.params:
        gflag = True
        self.params['group'] = 1
    axis_shape_flag = False
    if 'axis_shape' not in self.params:
        axis_shape_flag = True
        axis = self.params['axis']
        input_dim = self.inputs[0].betensor.dim()
        axis_positive = [ax + input_dim if ax < 0 else ax for ax in axis]
        axis_shape = [self.inputs[0].betensor.shape[ax] if ax in axis_positive else 1 for ax in range(input_dim)]
        self.params['axis_shape'] = axis_shape
        self.params['scale_shift_shape'] = [self.inputs[0].betensor.shape[ax]
                                            if ax == axis_positive[-1] else 1 for ax in range(input_dim)]
    groupnorm(self, *args)
    if aflag:
        self.params.pop('axis')
    if gflag:
        self.params.pop('group')
    if axis_shape_flag:
        self.params.pop('axis_shape')
        self.params.pop('scale_shift_shape')
