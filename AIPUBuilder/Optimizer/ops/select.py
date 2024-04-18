# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch


# @op_register(OpType.Select)
def select_forward(self, *args):
    inp0 = self.inputs[0]  # condition
    inp1 = self.inputs[1]  # x1
    inp2 = self.inputs[2]  # x2
    out = self.outputs[0]  # y
    condition = inp0.betensor
    x1 = inp1.betensor
    x2 = inp2.betensor
    if self.quantized:
        scale1, scale2 = self.params["scale_value"]
        shift1, shift2 = self.params["shift_value"]
        condition = condition.int() + inp0.zerop
        x1 = linear_requantize(x1 + inp1.zerop, scale1, shift1, out.zerop, out.qmin, out.qmax)
        x2 = linear_requantize(x2 + inp2.zerop, scale2, shift2, out.zerop, out.qmin, out.qmax)
    dim_num = max(max(x1.dim(), x2.dim()), condition.dim())
    for k in range(dim_num - x1.dim()):
        x1 = x1.unsqueeze(0)
    for k in range(dim_num - x2.dim()):
        x2 = x2.unsqueeze(0)
    for k in range(dim_num - condition.dim()):
        condition = condition.unsqueeze(0)
    x1_repeats = [1] * dim_num
    x2_repeats = [1] * dim_num
    cd_repeats = [1] * dim_num
    for k in range(dim_num):
        m = max(max(x1.shape[k], x2.shape[k]), condition.shape[k])
        x1_repeats[k] += m - x1.shape[k]
        x2_repeats[k] += m - x2.shape[k]
        cd_repeats[k] += m - condition.shape[k]
    if x1.dtype != x2.dtype:
        if dtype2bits(torch_type2dtype(x1.dtype)) >= dtype2bits(torch_type2dtype(x2.dtype)):
            x2 = x2.to(x1.dtype)
        else:
            x1 = x1.to(x2.dtype)
    y = torch.where(condition.bool().repeat(cd_repeats), x1.repeat(x1_repeats), x2.repeat(x2_repeats))
    out.betensor = y
    return out.betensor


# @quant_register(OpType.Select)
def select_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp0 = self.inputs[0]  # condition
    inp1 = self.inputs[1]  # x1
    inp2 = self.inputs[2]  # x2
    out = self.outputs[0]  # y
    if inp2.qinvariant != inp1.qinvariant:
        OPT_WARN(
            'Except for condition input, one input is quantize invariant and other one input is not, which may cause accuracy issue. layer_id=%s, %s' % (
                self.attrs['layer_id'], self.name),
            workflow_name='quantize', op_name=str(self.type))
    out_signed = is_signed(inp2.dtype) or is_signed(inp1.dtype)
    if inp2.qinvariant and inp1.qinvariant:
        out.scale = 1.0
        out.zerop = 0
        out.qbits = max(inp2.qbits, inp1.qbits)
        out.dtype = bits2dtype(out.qbits, is_signed=out_signed)
        out.qinvariant = True
        self.params["shift_value"] = [0, 0]
        self.params["shift_type"] = [SHIFT_DTYPE, SHIFT_DTYPE]
        self.params["scale_value"] = [1, 1]
        self.params["scale_type"] = [Dtype.UINT8, Dtype.UINT8]
    else:
        out.qbits = q_bits_activation
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(out,
                                                                                                  q_mode_activation,
                                                                                                  out.qbits,
                                                                                                  is_signed=out_signed)
        out.qinvariant = False

        local_rescale1 = out.scale / inp1.scale
        local_rescale2 = out.scale / inp2.scale
        inp_scale_max = max(inp2.scale, inp1.scale)
        if inp2.qinvariant and not inp1.qinvariant:
            local_rescale2 = 1.0
        if inp1.qinvariant and not inp2.qinvariant:
            local_rescale1 = 1.0
        do_scale1, do_scale1_type, do_shift1, do_shift1_type = \
            get_scale_approximation_params(local_rescale1, mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)
        do_scale2, do_scale2_type, do_shift2, do_shift2_type = \
            get_scale_approximation_params(local_rescale2, mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)
        self.params["shift_value"] = [int(do_shift1), int(do_shift2)]
        self.params["shift_type"] = [do_shift1_type, do_shift2_type]
        self.params["scale_value"] = [int(do_scale1), int(do_scale2)]
        self.params["scale_type"] = [do_scale1_type, do_scale2_type]
