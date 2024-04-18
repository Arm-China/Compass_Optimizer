# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch


register_optype('SufficientStatistics')


@quant_register(OpType.SufficientStatistics)
def sufficient_tatistics_quantized(self, *args):
    inp0, inp1 = self.inputs[0], self.inputs[1]
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("SufficientStatistics currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    if inp0.qinvariant != inp1.qinvariant:
        OPT_WARN(
            'one input is quantize invariant and other one input is not, which may cause accuracy issue. layer_id=%s, %s' % (
                self.attrs['layer_id'], self.name),
            workflow_name='quantize', op_name=str(self.type))
    if inp0.qbits != inp1.qbits:
        OPT_WARN(
            'qbits of two inputs are not equal , which may cause accuracy issue. better set cast_dtypes_for_lib=True in opt cfg. layer_id=%s, %s' % (
                self.attrs['layer_id'], self.name),
            workflow_name='quantize', op_name=str(self.type))

    out_signed_list = [True, False or self.force_dtype_int]
    if inp0.qinvariant and inp1.qinvariant:
        for idx, _ in enumerate(self.outputs):
            out = self.outputs[idx]
            out.scale = 1.0
            out.zerop = 0
            out.qbits = 32
            out.dtype = bits2dtype(out.qbits, is_signed=out_signed_list[idx])
            out.qinvariant = True
    else:
        for idx, _ in enumerate(self.outputs):
            out = self.outputs[idx]
            out.qinvariant = False
            out.qbits = q_bits_activation
            out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
                out, q_mode_activation, out.qbits, is_signed=out_signed_list[idx])

    inp0_qbits_max, _ = range2bits(inp0.qmin+inp0.zerop, inp0.qmax+inp0.zerop, force_int=True)
    inp1_qbits_max, _ = range2bits(inp1.qmin+inp1.zerop, inp1.qmax+inp1.zerop, force_int=True)
    reduce_multiplier = max(2 * (max(inp0_qbits_max, inp1_qbits_max) - 15), 1)
    left_shift = min(int(max(0, 15 - inp0_qbits_max)), int(max(0, 15 - inp1_qbits_max)))
    inp_scale_min = min(inp0.scale, inp1.scale) / reduce_multiplier
    input0__multiplier = inp_scale_min / inp0.scale * 2**left_shift
    input1__multiplier = inp_scale_min / inp1.scale * 2**left_shift
    output0__multiplier = self.outputs[0].scale / inp_scale_min / 2**(left_shift)
    output1__multiplier = self.outputs[1].scale / (inp_scale_min * inp_scale_min) / 2**(2 * left_shift)
    input0_scale, input0_scale_type, input0_shift, input0_shift_type = \
        get_scale_approximation_params(input0__multiplier, mult_bits=16,
                                       force_shift_positive=self.force_shift_positive)
    input1_scale, input1_scale_type, input1_shift, input1_shift_type = \
        get_scale_approximation_params(input1__multiplier, mult_bits=16,
                                       force_shift_positive=self.force_shift_positive)
    out0_scale, out0_scale_type, out0_shift, out0_shift_type = \
        get_scale_approximation_params(output0__multiplier, mult_bits=16,
                                       force_shift_positive=self.force_shift_positive)
    out1_scale, out1_scale_type, out1_shift, out1_shift_type = \
        get_scale_approximation_params(output1__multiplier, mult_bits=16,
                                       force_shift_positive=self.force_shift_positive)

    self.params["shift_value"] = [int(out0_shift), int(out1_shift), int(input0_shift), int(input1_shift)]
    self.params["shift_type"] = [out0_shift_type, out1_shift_type, input0_shift_type, input1_shift_type]
    self.params["scale_value"] = [int(out0_scale), int(out1_scale), int(input0_scale), int(input1_scale)]
    self.params["scale_type"] = [out0_scale_type, out1_scale_type, input0_scale_type, input1_scale_type]


@op_register(OpType.SufficientStatistics)
def sufficient_tatistics_forward(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out0 = self.outputs[0]
    out1 = self.outputs[1]

    x = inp0.betensor + (torch.tensor(0) if not self.quantized else torch.tensor(self.inputs[0].zerop))
    shift = inp1.betensor + (torch.tensor(0) if not self.quantized else torch.tensor(self.inputs[1].zerop))
    axis = self.get_param('axis')

    if self.quantized:
        out0_scale, out1_scale, input0_scale, input1_scale = self.params["scale_value"]
        out0_shift, out1_shift, input0_shift, input1_shift = self.params["shift_value"]

        # calculate mean
        act_qmin, act_qmax = -2 ** 31, 2 ** 31 - 1
        # In fact, the true value domain of x and shift after requantize is within the range of int15
        x = linear_requantize(x, input0_scale, input0_shift, 0, act_qmin, act_qmax).long()
        shift = linear_requantize(shift, input1_scale, input1_shift, 0, act_qmin, act_qmax).long()
        # In fact, the true value domain of m_ss is within the range of int16
        m_ss = x - shift
        output0 = torch.sum(m_ss, axis, keepdims=True)
        # Limit the number of items to 65535
        output0 = torch.clamp(output0, act_qmin, act_qmax)
        out0.betensor = torch.clamp(torch.div(output0 * out0_scale, 2**out0_shift,
                                    rounding_mode='trunc').int() - out0.zerop, out0.qmin, out0.qmax).long()

        v_ss = (m_ss * m_ss).long()
        v_ss = torch.clamp(v_ss, 0, 2 ** 32 - 1)
        high16_v_ss = torch.div(v_ss, 2 ** 16, rounding_mode='floor')
        high_sum = torch.sum(high16_v_ss, axis, keepdims=True)
        high_sum = torch.clamp(high_sum, 0, 2 ** 32 - 1)
        high_output = (high_sum * out1_scale * 2**16)

        low16_v_ss = v_ss - high16_v_ss * 2 ** 16
        low_sum = torch.sum(low16_v_ss, axis, keepdims=True).long()
        low_sum = torch.clamp(low_sum, 0, 2 ** 32 - 1)
        low_output = (low_sum * out1_scale)
        out1.betensor = torch.clamp(((high_output + low_output) >> out1_shift) - out1.zerop, out1.qmin, out1.qmax).int()
    else:
        m_ss = x - shift
        v_ss = m_ss * m_ss
        out0.betensor = torch.sum(m_ss, axis, keepdims=True)
        out1.betensor = torch.sum(v_ss, axis, keepdims=True)
    return out0.betensor, out1.betensor
