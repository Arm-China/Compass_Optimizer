# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils.passes_utils import passes_run


@passes_run
def shrink_pow_exponent(graph, config=None):
    def criteria(n):
        if n is not None and n.type == OpType.Pow:
            pow_parent = None
            for parent in n.parents:
                for outp in parent.outputs:
                    if outp.name == n.inputs[1].name:
                        pow_parent = parent
                        break
                if pow_parent:
                    break
            pow_nods, count_root, count_constant = pow_parent.get_ancestors()
            if count_root > 0 and count_root == count_constant:
                for node in pow_nods:
                    node.forward()
                unq = n.inputs[1].betensor.unique()
                if unq.numel() == 1:
                    return True
        return False
    # for powN: collect all inputs edge , if all of them == constant, then exponent should be N
    for n in graph.nodes:
        if criteria(n):
            unq = n.inputs[1].betensor.unique()
            n.params['exponent'] = float(unq[0])
