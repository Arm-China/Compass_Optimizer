# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils.quant_tool_utils import *
from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.framework import *

import torch


@op_register(OpType.PRELU)
# PReLU(x)=max(0,x)+weight∗min(0,x)
def prelu(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    negative_slope = self.constants['negative_slope'].betensor
    if len(negative_slope.shape) >= len(inp.ir_shape):
        piece = negative_slope.split(1)[0]
        if (((piece + torch.zeros_like(negative_slope)) - negative_slope).abs() < OPT_EPSILON).all().item():
            negative_slope = piece
        negative_slope = torch.squeeze(negative_slope, 0)
    zeros = torch.tensor(0, device=out.betensor.device)
    if self.quantized:
        x = inp.betensor + inp.zerop
        negative_slope_shift = self.get_param("negative_slope_shift")
        # here, need not clamp, to use linear_requantize, we using a bigest value in hardware platform
        tmp_qmin, tmp_qmax = bits2range(48, is_signed(out.dtype))
        y = linear_requantize(torch.minimum(x, zeros), negative_slope, negative_slope_shift, -
                              torch.maximum(x, zeros), tmp_qmin, tmp_qmax)
        # y = torch.maximum(x, zeros) + torch.round((negative_slope.int())*torch.minimum(x, zeros)*(0.5**negative_slope_shift))
        if 'shift_value' in self.params:
            do_shift = self.params["shift_value"]
            do_scale = self.params["scale_value"]
        elif 'shift' in self.constants:
            do_shift = self.constants["shift"].betensor
            do_scale = self.constants["scale"].betensor
        else:
            do_shift = 0
            do_scale = 1

        out.betensor = linear_requantize(y, do_scale, do_shift, out.zerop, out.qmin, out.qmax)
    else:
        # x = inp.betensor.permute(0,3,1,2)
        # out.betensor = torch.nn.functional.prelu(x,negative_slope).permute(0,2,3,1)
        out.betensor = torch.maximum(inp.betensor, zeros) + negative_slope*torch.minimum(inp.betensor, zeros)
    return out.betensor


@quant_register(OpType.PRELU)
def prelu_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]

    q_mode_activation = self.attrs["q_mode_activation"]

    q_bits_activation = self.attrs["q_bits_activation"]
    out.qinvariant = False
    out.qbits = q_bits_activation
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, True)
    do_scale, do_scale_type, do_shift, do_shift_type = \
        get_scale_approximation_params(out.scale / inp.scale, mult_bits=out.qbits,
                                       force_shift_positive=self.force_shift_positive)

    negative_slope_shift = self.attrs['scaling_bits'][0]
    negative_slope = linear_quantize_clip(
        self.constants['negative_slope'].betensor, (2 ** negative_slope_shift), 0, -32768, 32767)
    if len(negative_slope.shape) >= len(inp.ir_shape):
        piece = negative_slope.int().split(1)[0]
        # repeats = [negative_slope.shape[i] // piece.shape[i] for i in range(piece.dim())]
        # if 0 == torch.bitwise_xor(negative_slope.int(), piece.repeat(repeats)).sum().item() :
        if not torch.bitwise_xor(negative_slope.int(), piece).bool().any().item():
            negative_slope = piece
        negative_slope = torch.squeeze(negative_slope, 0)
    self.constants['negative_slope'].betensor = negative_slope
    self.constants['negative_slope'].dtype = bits2dtype(16, True)
    self.constants['negative_slope'].scale = 2 ** negative_slope_shift
    self.constants['negative_slope'].zerop = 0
    self.constants['negative_slope'].qbits = 16
    self.constants['negative_slope'].qinvariant = False
    self.params["negative_slope_shift"] = negative_slope_shift

    self.params["shift_value"] = int(do_shift)
    self.params["shift_type"] = do_shift_type
    self.params["scale_value"] = int(do_scale)
    self.params["scale_type"] = do_scale_type
