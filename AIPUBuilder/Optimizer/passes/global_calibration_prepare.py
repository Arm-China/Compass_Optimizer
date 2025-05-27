# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *


def global_calibration_prepare(graph: PyGraph, config):
    for method in config.global_calibration:
        mname = method[0]
        if 'smooth_quant_zy' == mname:
            for node in graph.nodes:
                if node.type in [OpType.FullyConnected, ]:
                    node.inputs[0].key_axis = len(node.inputs[0].ir_shape) - 1
        elif 'awq_zy' == mname:
            # def add_inp_abs_plh_for_fc(n: PyNode):
            #     inp_abs = n.inputs[0].betensor.abs().float()
            #     if len(n.placeholders) < 1:
            #         plh = PyTensor(n.name+'/inp_abs', dtype=Dtype.FP32)
            #         n.placeholders.append(plh)
            #     n.placeholders[0].betensor = inp_abs
            #     n.placeholders[0].key_axis = len(node.inputs[0].ir_shape) - 1
            for node in graph.nodes:
                if node.type in [OpType.FullyConnected, ]:
                    node.inputs[0].key_axis = len(node.inputs[0].ir_shape) - 1
                    # node.forward_hook = add_inp_abs_plh_for_fc
        else:
            pass
