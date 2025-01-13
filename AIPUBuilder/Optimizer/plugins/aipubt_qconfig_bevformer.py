# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
import numpy as np
import re


@register_plugin(PluginType.QConfig, '1.0')
class Bevformerqconfig(object):

    def __init__(self, graph, config):
        self.graph = graph
        self.config = config

    def _get_matched(self, keys, rule):
        matcheds = []
        for k in keys:
            m = re.match(rule, k)
            if m is not None:
                matcheds.append(m.group())
        return matcheds

    def __call__(self):
        # 8bits rule
        rules_for_8bits = [
            f"(.*)img(.*)",
            f"/img_backbone(.*)",
            f"/img_neck(.*)",
            f"/pts_bbox_head/transformer/Reshape_1",
            f"/img_neck/0/conv_1/Conv_post_transpose",
            f"/pts_bbox_head/transformer/Add_1(.*)",
            f"ADD",
            f"ADD_1",
            f"shift_0",
            f"pre_bev_0",
            f"/pts_bbox_head/transformer/Add_2_pre_tile"
        ]

        rules_for_weight_only = [
            OpType.FullyConnected,
        ]

        rules_for_fp16 = []

        for n in self.graph.nodes:
            n.attrs['trigger_float_op'] = 'float16_preferred'

            for r in rules_for_8bits:
                if '*' in r:
                    matched = self._get_matched([n.name], r)
                    if len(matched):
                        n.attrs['trigger_float_op'] = 'disable'
                        break
                else:
                    if n.name == r:
                        n.attrs['trigger_float_op'] = 'disable'
                        break

            # if 'encoder' not in n.name and 'decoder' not in n.name and int(n.attrs['layer_id']) > 906:
            #     n.attrs['trigger_float_op'] = 'disable'
            #     if n.type in [OpType.BatchNorm]:
            #         n.attrs['trigger_float_op'] = 'float16_preferred'

            # if 'encoder' in n.name:
            #     n.attrs['trigger_float_op'] = 'disable'

            for rn_type in rules_for_weight_only:
                if n.type == rn_type:
                    n.attrs['trigger_float_op'] = 'float16_act_int_wht'
                    break

            if n.type in [OpType.LayerNorm, OpType.Softmax]:
                n.attrs['trigger_float_op'] = 'float16_preferred'

            if n.type == OpType.Eltwise and len(n.children) == 1 and n.children[0].type == OpType.LayerNorm:
                n.attrs['trigger_float_op'] = 'float16_preferred'
