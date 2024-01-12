# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

import torch
import numpy as np
import pytest

import sys
import os
PKG_DIR = os.path.join(os.path.dirname(__file__), "../../../../")
sys.path.insert(0, PKG_DIR)

from AIPUBuilder.Optimizer.ops.atan import atan_forward, atan_quantize  # noqa
from AIPUBuilder.Optimizer.framework import *  # noqa
from AIPUBuilder.Optimizer.utils import *  # noqa


@pytest.mark.parametrize("range", [0.5, 1, 10, 100])
@pytest.mark.parametrize("q_mode", ['per_tensor_asymmetric', 'per_tensor_symmetric_full_range'])
@pytest.mark.parametrize("bits", [8, 16])
def test_atan(range, q_mode, bits):
    in_shape = [300, 1]
    out_shape = [300, 1]
    in_data = torch.randn(in_shape).to(torch.float32) * range

    atan_node = PyNode('atan', OpType.Atan)
    in_tensor = PyTensor('atan_in', in_data, Dtype.FP32)
    out_tensor = PyTensor('atan_out', TensorShape(out_shape), Dtype.FP32)
    atan_node.add_input(in_tensor)
    atan_node.add_output(out_tensor)

    OPT_INFO(f"{atan_node}")

    fp_out = atan_forward(atan_node)

    atan_node.inputs[0].min = atan_node.inputs[0].betensor.min().item()
    atan_node.inputs[0].max = atan_node.inputs[0].betensor.max().item()
    atan_node.outputs[0].min = atan_node.outputs[0].betensor.min().item()
    atan_node.outputs[0].max = atan_node.outputs[0].betensor.max().item()

    atan_node.attrs['q_mode_activation'] = q_mode
    atan_node.attrs['q_bits_activation'] = bits
    atan_node.attrs['lut_items_in_bits'] = 8 if bits <= 8 else 10

    in_tensor.scale, in_tensor.zerop, in_tensor.qmin, in_tensor.qmax, in_tensor.dtype = get_linear_quant_params_from_tensor(
        in_tensor, q_mode, bits, in_data.min() < 0)
    in_tensor.qbits = bits

    atan_quantize(atan_node)

    atan_node.quantized = True
    atan_node.inputs[0].betensor = linear_quantize_clip(atan_node.inputs[0].betensor,
                                                        in_tensor.scale,
                                                        in_tensor.zerop,
                                                        in_tensor.qmin,
                                                        in_tensor.qmax)
    q_out = atan_forward(atan_node)
    deq_out = (q_out + out_tensor.zerop) / out_tensor.scale

    cos = cosine_distance(fp_out, deq_out)
    MSE = torch.nn.MSELoss()
    mse = MSE(fp_out, deq_out)

    # print(f"cos={cos}, mse={mse},  and configure is [range={range}, q_mode={q_mode}, bits={bits}")
    # print(f"lut.size = {atan_node.constants['lut'].betensor.shape}, and lut = {atan_node.constants['lut'].betensor}")
    assert cos > 0.99, f"cos(={cos}) < 0.99, and configure is [range={range}, q_mode={q_mode}, bits={bits}"
    assert mse < 0.1, f"mse(={mse}) > 0.1, and configure is [range={range}, q_mode={q_mode}, bits={bits}"


if __name__ == '__main__':
    test_atan(100, 'per_tensor_asymmetric', 8)
