# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch

register_optype('Hardswish')


@quant_register(OpType.Hardswish)
def hardswish_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    out = self.outputs[0]
    out.qbits = q_bits_activation
    out_sign = True
    dev = inp.betensor.device
    # out.scale, out.zerop = 2**out.qbits - 1, 0
    # out.qmin, out.qmax = 0, 2**out.qbits - 1
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, out_sign)
    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut = linear_dequantize(torch.linspace(inp.qmin, inp.qmax, steps=lsteps, device=dev), inp.scale, inp.zerop)
    hard_swish_opt_tag = self.get_attrs('hard_swish_opt_tag', optional=True, default_value=0)
    if hard_swish_opt_tag != 0:
        lut = (lut - 3.0)
    lut = torch.nn.functional.hardswish(lut)
    if hard_swish_opt_tag == 2:
        lut = lut+0.375
    lut = linear_quantize_clip(lut, out.scale, out.zerop, out.qmin, out.qmax)
    self.constants["lut"] = PyTensor(self.name+"/hardswish_lut", lut.cpu().numpy().astype(dtype2nptype(out.dtype)))
    out.qinvariant = False


@op_register(OpType.Hardswish)
def hardswish(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    hard_swish_opt_tag = self.get_attrs('hard_swish_opt_tag', optional=True, default_value=0)
    if self.quantized:
        x = inp.betensor
        x = x - inp.qmin
        lut = self.constants["lut"].betensor
        x = torch.reshape(x, (-1,))
        y = lookup_lut_powerof2(x, lut, inp.qbits, False, dtype2bits(
            self.constants["lut"].dtype), is_signed(self.constants["lut"].dtype))
        out.betensor = torch.reshape(y, inp.betensor.shape)
    else:
        x = inp.betensor
        if hard_swish_opt_tag != 0:
            x = (x - 3.0)
        if hard_swish_opt_tag == 2:
            out.betensor = torch.nn.functional.hardswish(x)+0.375
        else:
            out.betensor = torch.nn.functional.hardswish(x)
    return out.betensor
