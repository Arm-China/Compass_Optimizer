# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.ops.relu import *
from AIPUBuilder.Optimizer.ops.clip import *


def relu_family(self, *args):
    if {'max_clamp_value', 'min_clamp_value'}.issubset(self.params.keys()):
        return clip(self, *args)
    else:
        return relu(self, *args)


def relu_family_quantize(self, *args):
    if {'max_clamp_value', 'min_clamp_value'}.issubset(self.params.keys()):
        clip_quantize(self, *args)
    else:
        relu_quantize(self, *args)


def relu_family_out_signed(self, *args):
    if {'max_clamp_value', 'min_clamp_value'}.issubset(self.params.keys()):
        clip_out_signed(self, *args)
    else:
        return False
