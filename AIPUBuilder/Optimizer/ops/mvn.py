# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.groupnorm import groupnorm_quantize, groupnorm
import torch

mvn_norm_prefix = ['scale', 'shift']
other_norm_prefix = ['norm_scale', 'norm_shift']
suffix = ['_value', '_type']


def update_name(node, source_list, dst_list):
    for idx, v in enumerate(source_list):
        for suf in suffix:
            source_name = source_list[idx] + suf
            dst_name = dst_list[idx] + suf
            node.params.update({dst_name: node.params.pop(source_name)})


@quant_register(OpType.MVN)
def mvn_quantize(self, *args):
    gflag = False
    if 'group' not in self.params:
        gflag = True
        self.params['group'] = 1
    groupnorm_quantize(self, *args)
    if gflag:
        self.params.pop('group')
    # update ngamma_scale to sqrt_scale for new IR
    update_name(self, other_norm_prefix, mvn_norm_prefix)


@op_register(OpType.MVN)
def mvn(self, *args):
    gflag = False
    if 'group' not in self.params:
        gflag = True
        self.params['group'] = 1
    if 'ngamma_scale_value' not in self.params and self.quantized:
        update_name(self, mvn_norm_prefix, other_norm_prefix)
        self.params['scale_value'] = 1
        self.params['shift_value'] = 0
    groupnorm(self, *args)
    if gflag:
        self.params.pop('group')
    if self.quantized:
        update_name(self, other_norm_prefix, mvn_norm_prefix)
        # self.params.update({'sqrt_scale_value' : self.params.pop('ngamma_scale_value')})
        # self.params.update({'sqrt_scale_type' : self.params.pop('ngamma_scale_type')})
        # self.params.update({'sqrt_shift_value' : self.params.pop('ngamma_shift_value')})
        # self.params.update({'sqrt_shift_type' : self.params.pop('ngamma_shift_type')})
        # self.params.update({'sqrt_zp_value' : self.params.pop('ngamma_zp_value')})
        # self.params.update({'sqrt_zp_type' : self.params.pop('ngamma_zp_type')})

        # self.params.update({'scale_value' : self.params.pop('norm_scale_value')})
        # self.params.update({'scale_type' : self.params.pop('norm_scale_type')})
        # self.params.update({'shift_value' : self.params.pop('norm_shift_value')})
        # self.params.update({'shift_type' : self.params.pop('norm_shift_type')})
