# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch
import math


#  0.5 * x * (1 + tanh ( sqrt(2/pi) * (x + 0.044715 * x**3)  ) )
#  0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gelu_tanh_apprx(x):
    #  increase compatibility for 3.6.5(no torch.pi api)
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / math.pi)) * (x + 0.044715 * torch.pow(x, 3))))


@quant_register(OpType.GELU)
def gelu_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]
    apprx = self.get_param('approximate').lower()  # NONE, Tanh

    inp = self.inputs[0]
    out = self.outputs[0]
    out.qbits = q_bits_activation
    out_sign = True
    dev = inp.betensor.device
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, out_sign)
    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut = linear_dequantize(torch.linspace(inp.qmin, inp.qmax, steps=lsteps, device=dev), inp.scale, inp.zerop)

    if apprx == 'none':
        torch_gelu = torch.nn.GELU()
        lut = torch_gelu(lut)
    elif apprx == 'tanh':
        lut = gelu_tanh_apprx(lut)

    lut = linear_quantize_clip(lut, out.scale, out.zerop, out.qmin, out.qmax)
    self.constants["lut"] = PyTensor(self.name + "/gelu_lut", lut.cpu().numpy().astype(dtype2nptype(out.dtype)))
    out.qinvariant = False


@op_register(OpType.GELU)
def gelu(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    apprx = self.get_param('approximate').lower()  # NONE, Tanh
    if apprx not in ['none', 'tanh']:
        OPT_ERROR('GELU dont support approximation method:%s' % apprx)

    if self.quantized:
        x = inp.betensor
        x = x - inp.qmin
        lut = self.constants["lut"].betensor
        x = torch.reshape(x, (-1,))
        y = lookup_lut_powerof2(x, lut, inp.qbits, False, dtype2bits(
            self.constants["lut"].dtype), is_signed(self.constants["lut"].dtype))
        out.betensor = torch.reshape(y, inp.betensor.shape)

    else:

        if apprx == 'none':
            out.betensor = torch.nn.functional.gelu(inp.betensor)
        elif apprx == 'tanh':
            out.betensor = gelu_tanh_apprx(inp.betensor)
    return out.betensor
