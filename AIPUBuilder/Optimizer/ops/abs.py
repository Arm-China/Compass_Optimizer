# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.ops.cast import forward_with_clip
from AIPUBuilder.Optimizer.utils import *


@op_register(OpType.Abs)
def Abs_forward(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    if self.quantized:
        if 'scale_value' in self.params:
            shift = self.params["shift_value"]
            scale = self.params["scale_value"]
            value = inp.betensor.int() + inp.zerop
            value = torch.abs(value).int()
            out.betensor = linear_requantize(value, scale, shift, out.zerop, out.qmin, out.qmax)
        else:
            value = inp.betensor.long()
            value = torch.abs(value).long()
            out.betensor = forward_with_clip(value, out.dtype, 'TRUNCATION')
    else:
        value = torch.abs(inp.betensor)
        out.betensor = forward_with_clip(value, out.dtype, 'TRUNCATION')
    return out.betensor


@quant_register(OpType.Abs)
def abs_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    q_mode_activation = self.attrs["q_mode_activation"]
    q_bits_activation = self.attrs["q_bits_activation"]

    multiplier_bits = min(inp.qbits, 16)
    if inp.qinvariant:
        out.qinvariant = True
        out.scale = inp.scale
        out.zerop = inp.zerop
        out.dtype = inp.dtype
        out.qbits = inp.qbits
        out.qmin, out.qmax = dtype2range(out.dtype)
    else:
        out.qinvariant = False
        out_signed = False or self.force_dtype_int
        out.qbits = q_bits_activation
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, out.qbits, is_signed=out_signed)
        # default scale/shift
        self.params["shift_value"] = 0
        self.params["shift_type"] = Dtype.INT8
        self.params["scale_value"] = 1
        self.params["scale_type"] = bits2dtype(multiplier_bits, False)
        if out.scale != inp.scale:
            scale, scale_type, shift, shift_type = \
                get_scale_approximation_params(out.scale / inp.scale, mult_bits=multiplier_bits,
                                               force_shift_positive=self.force_shift_positive)
            self.params["shift_value"] = int(shift)
            self.params["shift_type"] = shift_type
            self.params["scale_value"] = int(scale)
            self.params["scale_type"] = scale_type
