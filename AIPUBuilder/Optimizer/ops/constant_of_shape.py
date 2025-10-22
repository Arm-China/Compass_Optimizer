# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import torch

register_optype('ConstantOfShape')


@op_register(OpType.ConstantOfShape)
def constant_of_shape_forward(self, *args):
    inp = self.inputs[0].betensor
    value = self.get_param('value')
    out = torch.full(inp.tolist(), value)
    self.outputs[0].betensor = out
    return out


@quant_register(OpType.ConstantOfShape)
def constant_of_quantize(self, *args):
    out = self.outputs[0]
    q_mode_activation = QuantMode.to_per_tensor(self.attrs["q_mode_activation"])
    q_bits_activation = self.attrs["q_bits_activation"]
    value = self.get_param('value')
    vt = torch.tensor(value, device=out.betensor.device)
    out_signed = value < 0 or self.force_dtype_int
    out.qbits = q_bits_activation
    if torch.isnan(vt).any() or torch.isinf(vt).any() or (vt.ceil() - vt.floor()).max() == 0:
        out.scale = 1.
        out.zerop = 0
        out.dtype = bits2dtype(out.qbits, is_signed=out_signed)
        out.qmin, out.qmax = dtype2range(out.dtype)
        out.qinvariant = True
    else:
        out.max = abs(value)
        out.min = -out.max
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, out.qbits, is_signed=out_signed)
        out.qinvariant = False
    self.params['value'] = linear_quantize_clip(vt, out.scale, out.zerop, out.qmin, out.qmax).item()
