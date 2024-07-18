# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch


@quant_register(OpType.SquaredDifference)
def squareddifference_quantizes(self, *args):
    inp0, inp1 = self.inputs[0], self.inputs[1]
    out = self.outputs[0]
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation):
        OPT_FATAL("SquaredDifference currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    out_signed = False or self.force_dtype_int
    if inp0.qinvariant != inp1.qinvariant:
        OPT_WARN(f'{self}, one input is quantize invariant and other one input is not, which may cause accuracy issue.',
                 workflow_name='quantize')
    if inp0.qbits != inp1.qbits:
        OPT_WARN(f"{self}, qbits of two inputs are not equal , which may cause accuracy issue. better set "
                 f"cast_dtypes_for_lib=True in opt cfg", workflow_name='quantize')

    if inp0.qinvariant and inp1.qinvariant:
        out.scale = 1.0
        out.zerop = 0
        out.qbits = 32
        out.dtype = bits2dtype(out.qbits, is_signed=out_signed)
        out.qmin, out.qmax = dtype2range(out.dtype)
        out.qinvariant = True

        self.params["shift_value"] = [0, 0, 0]
        self.params["shift_type"] = [Dtype.INT8, Dtype.INT8, Dtype.INT8]
        self.params["scale_value"] = [1, 1, 1]
        self.params["scale_type"] = [Dtype.UINT16, Dtype.UINT16, Dtype.UINT16]
    else:
        out.qinvariant = False
        out.qbits = q_bits_activation
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, out.qbits, is_signed=out_signed)

        oscale = out.scale
        inp0_qbits_max, _ = range2bits(inp0.qmin+inp0.zerop, inp0.qmax+inp0.zerop, force_int=True)
        inp1_qbits_max, _ = range2bits(inp1.qmin+inp1.zerop, inp1.qmax+inp1.zerop, force_int=True)
        reduce_multiplier = max(2 * (max(inp0_qbits_max, inp1_qbits_max) - 15), 1)
        left_shift = min(int(max(0, 15 - inp0_qbits_max)), int(max(0, 15 - inp1_qbits_max)))
        inp_scale_min = min(inp0.scale, inp1.scale) / reduce_multiplier
        input0__multiplier = inp_scale_min / inp0.scale * 2**left_shift
        input1__multiplier = inp_scale_min / inp1.scale * 2**left_shift
        output__multiplier = oscale / (inp_scale_min * inp_scale_min) / 2**(2 * left_shift)
        input0_scale, input0_scale_type, input0_shift, input0_shift_type = \
            get_scale_approximation_params(input0__multiplier, mult_bits=16,
                                           force_shift_positive=self.force_shift_positive)
        input1_scale, input1_scale_type, input1_shift, input1_shift_type = \
            get_scale_approximation_params(input1__multiplier, mult_bits=16,
                                           force_shift_positive=self.force_shift_positive)
        out_scale, out_scale_type, out_shift, out_shift_type = \
            get_scale_approximation_params(output__multiplier, mult_bits=16,
                                           force_shift_positive=self.force_shift_positive)

        self.params["shift_value"] = [int(out_shift), int(input0_shift), int(input1_shift)]
        self.params["shift_type"] = [out_shift_type, input0_shift_type, input1_shift_type]
        self.params["scale_value"] = [int(out_scale), int(input0_scale), int(input1_scale)]
        self.params["scale_type"] = [out_scale_type, input0_scale_type, input1_scale_type]


@op_register(OpType.SquaredDifference)
def squareddifference(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]

    x0 = inp0.betensor + (0 if not self.quantized else self.inputs[0].broadcast_zerop)
    x1 = inp1.betensor + (0 if not self.quantized else self.inputs[1].broadcast_zerop)

    if self.quantized:
        out_scale, input0_scale, input1_scale = self.params["scale_value"]
        out_shift, input0_shift, input1_shift = self.params["shift_value"]

        act_qmin, act_qmax = -2 ** 31, 2 ** 31 - 1
        # In fact, the true value domain of x0 and x1 after requantize is within the range of int15
        x0 = linear_requantize(x0, input0_scale, input0_shift, 0, act_qmin, act_qmax).long()
        x1 = linear_requantize(x1, input1_scale, input1_shift, 0, act_qmin, act_qmax).long()
        diff = x0 - x1
        squared_diff = (diff * diff).long()
        # diff is in the range of int16, so the result after square is in the range of 32bit
        squared_diff = torch.clamp(squared_diff, act_qmin, act_qmax)
        out.betensor = linear_requantize(squared_diff, out_scale, out_shift, out.zerop, out.qmin, out.qmax).long()
    else:
        minus = x0 - x1
        out.betensor = minus * minus
    return out.betensor
