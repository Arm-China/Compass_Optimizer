# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *


def get_ancestors(node):
    ancestors = []
    visited = {}

    def traverse_parents(node):
        if node:
            if len(node.parents) < 1:
                ancestors.append(node)
            else:
                for nparent in node.parents:
                    traverse_parents(nparent)
            if node.name not in visited:
                node.forward()  # get node's outputs by the way
                visited[node.name] = True
        return
    traverse_parents(node)
    return ancestors


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
        pow_nods = get_ancestors(pow_parent)
        count_constant = 0
        for itm in pow_nods:
            if itm.type == OpType.Constant:
                count_constant += 1
        if count_constant > 0 and count_constant == len(pow_nods):
            unq = n.inputs[1].betensor.unique()
            if unq.numel() == 1:
                return True
    return False


def shrink_pow_exponent_s1(graph, config):
    # for powN: collect all inputs edge , if all of them == constant, then exponent should be N
    for n in graph.nodes:
        if criteria(n):
            unq = n.inputs[1].betensor.unique()
            n.params['exponent'] = float(unq[0])
