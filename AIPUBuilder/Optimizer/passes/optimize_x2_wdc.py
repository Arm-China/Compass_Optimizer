# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *


def optimize_x2_wdc(graph: PyGraph, config=None):
    for node in graph.nodes:
        if not str(node.attrs['optimize_wdc_for_x2']).lower() == 'true':
            continue
        q_mode_weight = node.attrs['q_mode_weight']
        q_bits_weight = node.attrs['q_bits_weight']
        for k, w in node.constants.items():
            for trial in range(4):
                w.scale, w.zerop, w.qmin, w.qmax, w.dtype =\
                    get_linear_quant_params_from_tensor(w,
                                                        q_mode_weight, q_bits_weight, is_signed=True)
                # We only try to scale up weight 4 times to keep acc
                fake_quant = linear_quantize_clip(w.betensor, w.broadcast_scale, w.broadcast_zerop, w.qmin, w.qmax)
                comp_rate = simulate_x2_wdc(fake_quant, q_bits_weight)
                if comp_rate < 0.9:
                    OPT_DEBUG(f"weight {k} gets comp rate of {comp_rate} at step {trial}")
                    break
                else:
                    w.min *= 2
                    w.max *= 2
                    OPT_INFO(f"Scaling up {node.name}'s tensor {k} by 2 times to adapt WDC, acc may be affacted")
