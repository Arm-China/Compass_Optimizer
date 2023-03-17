# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.groupnorm import groupnorm_quantize, groupnorm
import torch


@quant_register(OpType.InstanceNorm)
def instancenorm_quantize(self, *args):
    groupnorm_quantize(self, *args)


@op_register(OpType.InstanceNorm)
def instancenorm(self, *args):
    aflag = False
    input_dim = self.inputs[0].betensor.dim()
    if 'axis' not in self.params:
        aflag = True
        # inp0.shape = [N,D1,D2,....,C]
        axis_param = [axis for axis in range(1, input_dim - 1)]
        self.params['axis'] = axis_param
    gflag = False
    if 'group' not in self.params:
        gflag = True
        self.params['group'] = 1
    axis_shape_flag = False
    if 'axis_shape' not in self.params:
        axis_shape_flag = True
        axis_shape = [1 for ax in range(input_dim-1)] + [self.inputs[0].betensor.shape[input_dim-1]]
        self.params['axis_shape'] = axis_shape
        self.params['scale_shift_shape'] = axis_shape
    groupnorm(self, *args)
    if aflag:
        self.params.pop('axis')
    if gflag:
        self.params.pop('group')
    if axis_shape_flag:
        self.params.pop('axis_shape')
        self.params.pop('scale_shift_shape')
