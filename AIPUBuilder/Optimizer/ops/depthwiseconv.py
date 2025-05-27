# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.ops.conv import conv2d, conv2d_quantize


@op_register(OpType.DepthwiseConv)
def depthwise_conv2d(self, *args):
    x = conv2d(self, *args)
    return x


@quant_register(OpType.DepthwiseConv)
def depthwise_conv2d_quantize(self, *args):
    conv2d_quantize(self, *args)
    if 'scale_value' in self.params and 'shift_value' in self.params and 'scale_type' in self.params and 'shift_type' in self.params:
        self.constants['scale'] = PyTensor(f"{self.name}_scale", [self.params['scale_value'], ]
                                           * self.constants['weights'].ir_shape[0], dtype=self.params['scale_type'])
        self.constants['shift'] = PyTensor(f"{self.name}_shift", [self.params['shift_value'], ]
                                           * self.constants['weights'].ir_shape[0], dtype=self.params['shift_type'])
        self.params.pop('scale_value')
        self.params.pop('shift_value')
        self.params.pop('scale_type')
        self.params.pop('shift_type')
