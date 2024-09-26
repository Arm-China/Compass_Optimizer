# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import PyGraph, PyNode, PyTensor, OpType, Dtype, TensorShape
import numpy as np
import torch


def test1():
    in_n = PyNode('in', OpType.Input)
    in_ot = PyTensor('in_t', np.random.randint(0, 255, [1, 1000]).astype(np.uint8))
    in_n.add_output(in_ot)
    in_ot.scale = 9.0
    in_ot.zerop = 1

    deq_n = PyNode('de', OpType.DeQuantize)
    deq_n.add_input(in_ot)
    deq_ot = PyTensor('deq_t', TensorShape([1, 1000]), Dtype.FP16)
    deq_n.add_output(deq_ot)

    q_n = PyNode('q', OpType.Quantize)
    q_n.add_input(deq_ot)
    q_ot = PyTensor('q_t', TensorShape([1, 1000]), Dtype.UINT8)
    q_n.add_output(q_ot)

    r_n = PyNode('r', OpType.Reshape)
    r_n.add_input(q_ot)
    r_ot = PyTensor('r_t', TensorShape([1000]), Dtype.UINT8)
    r_n.add_output(r_ot)

    g = PyGraph('test')
    g.add_node(in_n)
    g.add_node(deq_n)
    g.add_node(q_n)
    g.add_node(r_n)

    from AIPUBuilder.Optimizer.passes.merge_inserted_op import merge_inserted_op

    g.serialize('./test0.txt', './test0.bin')
    merge_inserted_op(g)
    g.serialize('./test.txt', './test.bin')


def test2():
    in_n = PyNode('in', OpType.Input)
    in_ot = PyTensor('in_t', np.random.rand(1, 1000).astype(np.float16))
    in_n.add_output(in_ot)

    q_n = PyNode('q', OpType.Quantize)
    q_n.add_input(in_ot)
    q_ot = PyTensor('q_t', TensorShape([1, 1000]), Dtype.UINT8)
    q_n.add_output(q_ot)

    deq_n = PyNode('de', OpType.DeQuantize)
    deq_n.add_input(q_ot)
    deq_ot = PyTensor('deq_t', TensorShape([1, 1000]), Dtype.FP16)
    deq_n.add_output(deq_ot)

    r_n = PyNode('r', OpType.Reshape)
    r_n.add_input(deq_ot)
    r_ot = PyTensor('r_t', TensorShape([1000]), Dtype.FP16)
    r_n.add_output(r_ot)

    g = PyGraph('test')
    g.add_node(in_n)
    g.add_node(deq_n)
    g.add_node(q_n)
    g.add_node(r_n)

    from AIPUBuilder.Optimizer.passes.merge_inserted_op import merge_inserted_op

    g.serialize('./test0.txt', './test0.bin')
    merge_inserted_op(g)
    g.serialize('./test1.txt', './test1.bin')


def test3():
    in_n = PyNode('in', OpType.Input)
    in_ot = PyTensor('in_t', np.random.rand(1, 1000).astype(np.float16))
    in_n.add_output(in_ot)

    q_n = PyNode('q', OpType.Quantize)
    q_n.add_input(in_ot)
    q_ot = PyTensor('q_t', TensorShape([1, 1000]), Dtype.UINT8)
    q_n.add_output(q_ot)

    c_n = PyNode('c', OpType.Cast)
    c_n.add_input(q_ot)
    c_ot = PyTensor('c_t', TensorShape([1, 1000]), Dtype.UINT16)
    c_n.add_output(c_ot)

    r_n = PyNode('r', OpType.Reshape)
    r_n.add_input(c_ot)
    r_ot = PyTensor('r_t', TensorShape([1, 1, 1000]), Dtype.FP16)
    r_n.add_output(r_ot)

    deq_n = PyNode('de', OpType.DeQuantize)
    deq_n.add_input(r_ot)
    deq_ot = PyTensor('deq_t', TensorShape([1, 1, 1000]), Dtype.FP16)
    deq_n.add_output(deq_ot)

    r_n1 = PyNode('r1', OpType.Reshape)
    r_n1.add_input(deq_ot)
    r_ot1 = PyTensor('r1_t', TensorShape([1000]), Dtype.FP16)
    r_n1.add_output(r_ot1)

    g = PyGraph('test')
    g.add_node(in_n)
    g.add_node(q_n)
    g.add_node(c_n)
    g.add_node(r_n)
    g.add_node(deq_n)
    g.add_node(r_n1)

    from AIPUBuilder.Optimizer.passes.merge_inserted_op import merge_inserted_op

    g.serialize('./test0.txt', './test0.bin')
    merge_inserted_op(g)
    g.serialize('./test1.txt', './test1.bin')


def test4():
    in_n = PyNode('in', OpType.Input)
    in_ot = PyTensor('in_t', np.random.rand(1, 1000).astype(np.uint16))
    in_n.add_output(in_ot)

    c_n = PyNode('c', OpType.Cast)
    c_n.add_input(in_ot)
    c_ot = PyTensor('c_t', TensorShape([1, 1000]), Dtype.INT32)
    c_n.add_output(c_ot)

    deq_n = PyNode('de', OpType.DeQuantize)
    deq_n.add_input(c_ot)
    deq_ot = PyTensor('deq_t', TensorShape([1, 1, 1000]), Dtype.FP16)
    deq_n.add_output(deq_ot)

    q_n = PyNode('q', OpType.Quantize)
    q_n.add_input(deq_ot)
    q_ot = PyTensor('q_t', TensorShape([1, 1000]), Dtype.UINT8)
    q_n.add_output(q_ot)

    r_n = PyNode('r', OpType.Reshape)
    r_n.add_input(q_ot)
    r_ot = PyTensor('r_t', TensorShape([1, 1, 1000]), Dtype.UINT8)
    r_n.add_output(r_ot)

    g = PyGraph('test')
    g.add_node(in_n)
    g.add_node(c_n)
    g.add_node(deq_n)
    g.add_node(q_n)
    g.add_node(r_n)

    from AIPUBuilder.Optimizer.passes.merge_inserted_op import merge_inserted_op

    g.serialize('./test0.txt', './test0.bin')
    merge_inserted_op(g)
    g.serialize('./test1.txt', './test1.bin')


if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    test4()
