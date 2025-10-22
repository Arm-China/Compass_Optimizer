# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
import numpy as np
import re


@register_plugin(PluginType.QConfig, '1.0')
class Qwen15_w4a8_qconfig(object):

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

        ]
        rules_for_16bits = [

        ]
        rules_for_float_op = [
            f"fully_connected_0",
            f"fully_connected_195",
        ]
        for n in self.graph.nodes:
            # n.attrs['trigger_float_op'] = 'float16_preferred'
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

            for r in rules_for_16bits:
                if '*' in r:
                    matched = self._get_matched([n.name], r)
                    if len(matched):
                        n.attrs['trigger_float_op'] = 'disable'
                        n.attrs['trigger_float_op'] = 'disable'
                        n.attrs['q_bits_weight'] = 16
                        n.attrs['q_bits_activation'] = 16
                        n.attrs['q_bits_bias'] = 40
                        break
                else:
                    if n.name == r:
                        n.attrs['trigger_float_op'] = 'disable'
                        n.attrs['q_bits_weight'] = 16
                        n.attrs['q_bits_activation'] = 16
                        n.attrs['q_bits_bias'] = 40
                        break

            for r in rules_for_float_op:
                if '*' in r:
                    matched = self._get_matched([n.name], r)
                    if len(matched):
                        n.attrs['trigger_float_op'] = 'float16_preferred'
                        break
                else:
                    if n.name == r:
                        n.attrs['trigger_float_op'] = 'float16_preferred'
                        break

            if n.type in [OpType.LayerNorm, OpType.Softmax]:
                n.attrs['trigger_float_op'] = 'float16_preferred'

            if n.type == OpType.Eltwise and len(n.children) == 1 and n.children[0].type == OpType.LayerNorm:
                n.attrs['trigger_float_op'] = 'float16_preferred'
