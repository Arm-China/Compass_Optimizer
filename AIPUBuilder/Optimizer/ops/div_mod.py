# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch


@quant_register(OpType.DivMod)
def div_mod_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out0 = self.outputs[0]
    out1 = self.outputs[1]

    if not inp0.qinvariant or not inp1.qinvariant:
        OPT_FATAL("the inputs of op{%s} should be quantization invariant." % (self.type))
    else:  # inp0.qinvariant is true and inp1.qinvariant is true
        if inp0.ir_dtype != inp1.ir_dtype:
            OPT_WARN(
                "currently the two inputs dtype of op{%s} is different, the result may not be as expected ." % (self.type))
        out_dtype = inp0.ir_dtype
        out_bits = dtype2bits(out_dtype)

        out0.qbits = out_bits
        out0.scale = 1.0
        out0.zerop = 0
        out0.dtype = out_dtype
        qmin, qmax = dtype2range(out_dtype)
        out0.qmin = qmin
        out0.qmax = qmax
        out0.qinvariant = True

        out1.qbits = out_bits
        out1.scale = 1.0
        out1.zerop = 0
        out1.dtype = out_dtype
        out1.qmin = qmin
        out1.qmax = qmax
        out1.qinvariant = True


@op_register(OpType.DivMod)
def div_mod_forward(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out0 = self.outputs[0]
    out1 = self.outputs[1]
    mode = self.get_param('mode', optional=True, default_value='FLOOR').lower()
    if mode not in ['floor', 'trunc']:
        OPT_FATAL("currently the op{%s} only supprot mode[floor,trunc], but now mode is {%s}." % (self.type, mode))

    torch_out_dtype = dtype2torch_type(inp0.ir_dtype)

    input0 = inp0.betensor.long()
    input1 = inp1.betensor.long()
    input0, input1 = broadcasting_transform(input0, input1)
    out0.betensor = torch.zeros_like(input0, device=inp0.betensor.device).to(torch_out_dtype)
    out1.betensor = torch.zeros_like(input0, device=inp0.betensor.device).to(torch_out_dtype)

    nonzeros_mask = input1 != 0
    quotient = torch.div(input0[nonzeros_mask], input1[nonzeros_mask], rounding_mode=mode)
    remainder = input0[nonzeros_mask] - input1[nonzeros_mask] * quotient

    quotient_np = quotient.cpu().numpy().astype(dtype2nptype(inp0.ir_dtype))
    pytensor = PyTensor('out0', quotient_np)
    out0.betensor[nonzeros_mask] = pytensor.betensor.to(inp0.betensor.device).to(torch_out_dtype)

    remainder_np = remainder.cpu().numpy().astype(dtype2nptype(inp0.ir_dtype))
    pytensor = PyTensor('out1', remainder_np)
    out1.betensor[nonzeros_mask] = pytensor.betensor.to(inp0.betensor.device).to(torch_out_dtype)

    return out0.betensor, out1.betensor
