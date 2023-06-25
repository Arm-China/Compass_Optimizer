# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *
from queue import Queue


def get_ancestors(node):
    q = Queue(maxsize=0)
    # total parent path node
    ancestors = []
    visited = {}

    def traverse_parents(node):
        count_root = 0
        count_constant = 0
        if node:
            q.put(node)
            ancestors.append(node)
            visited[node.name] = True

            while(q.qsize()):
                current_node = q.get()
                # get count_root and count_constant by the way
                if len(current_node.parents) < 1:
                    count_root += 1
                if current_node.type == OpType.Constant:
                    count_constant += 1
                for nparent in current_node.parents:
                    if nparent.name not in visited:
                        visited[nparent.name] = True
                        q.put(nparent)
                        ancestors.insert(0, nparent)
        return count_root, count_constant
    count_root, count_constant = traverse_parents(node)
    return ancestors, count_root, count_constant


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
        pow_nods, count_root, count_constant = get_ancestors(pow_parent)
        if count_root > 0 and count_root == count_constant:
            for node in pow_nods:
                node.forward()
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
