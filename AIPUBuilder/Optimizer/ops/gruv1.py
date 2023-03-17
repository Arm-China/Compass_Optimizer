# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.gruv3 import gruv3_quantize, gruv3
from AIPUBuilder.Optimizer.logger import *
import torch.nn as nn


@op_register(OpType.GRUv1)
def gruv1(self, *args):
    self.params['version'] = "GRUV1"
    gruv3(self, *args)
    self.params.pop('version')


@quant_register(OpType.GRUv1)
def gruv1_quantize(self, *args):
    self.params['version'] = "GRUV1"
    gruv3_quantize(self, *args)
    self.params.pop('version')
