# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import (
    PyGraph,
    PyNode,
    PyTensor,
    Dtype,
    OpType,
    TensorShape,
)
import torch


torch.manual_seed(42)


def test0():
    const_n1 = PyNode("i0", OpType.Constant)
    w0 = torch.rand([10000]).reshape(1, 1, 1, -1).float().cuda()
    w0_tile = w0.repeat([2, 3, 4, 1000])
    wt = PyTensor("w0", w0_tile)
    ot = PyTensor("o0", TensorShape(list(w0_tile.shape)), Dtype.FP32)
    const_n1.add_output(ot)
    const_n1.constants["weights"] = wt
    import time
    bt = time.time()
    x = wt.detile_betensor()
    et = time.time()
    print(f"cost time = {et - bt}")
    assert torch.equal(x, w0)

    x0 = wt.detile_betensor()
    eet = time.time()
    print(f"second cost time = {eet - et}")
    assert torch.equal(x0, w0)

    # const_n2 = PyNode("i1", OpType.Constant)
    # w1 = torch.rand([6]).reshape(1, 1, 1, -1).float().cuda()
    # w1_tile = w1.repeat([2, 3, 4, 1])
    # wt1 = PyTensor("w1", w1_tile)
    # ot1 = PyTensor("o1", TensorShape(list(w1_tile.shape)), Dtype.FP32)
    # const_n1.add_output(ot1)
    # const_n1.constants["weights"] = wt1

    # assert torch.equal(wt1.detile_betensor(), w1)


def test():
    # g = PyGraph('test')
    input_n1 = PyNode("i0", OpType.Constant)
    ot = PyTensor("o0", TensorShape([2, 3, 4, 6]), Dtype.FP32)
    input_n1.add_output(ot)

    const_n1 = PyNode("i1", OpType.Constant)
    w1 = torch.rand([6]).reshape(1, 1, 1, -1).float().cuda()
    w1_tile = w1.repeat([2, 3, 4, 1])
    wt1 = PyTensor("w1", w1_tile)
    ot1 = PyTensor("o1", TensorShape(list(w1_tile.shape)), Dtype.FP32)
    const_n1.add_output(ot1)
    const_n1.constants["weights"] = wt1
    ot1.betensor = w1_tile

    elt_n = PyNode("elt", OpType.Eltwise)
    elt_n.add_input(ot)
    elt_n.add_input(ot1)

    elt_ot = PyTensor("eltot", wt1.ir_shape, Dtype.FP32)
    elt_n.add_output(elt_ot)
    elt_n.params["method"] = "ADD"
    elt_n.params["with_activation"] = "NONE"

    from AIPUBuilder.Optimizer.ops.eltwise import eltwise

    ot1.pnode = const_n1
    idata = torch.rand([1, 1, 1, 6]).float().cuda()
    ot.betensor = idata

    o = eltwise(elt_n)
    assert torch.equal(o, idata + w1)


if __name__ == "__main__":
    test0()
    # test()
