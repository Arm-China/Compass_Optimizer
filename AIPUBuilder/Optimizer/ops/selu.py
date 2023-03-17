# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch


register_optype('SELU')
# SELU(x)=scale∗(max(0,x)+min(0,α∗(exp(x)−1)))
#α = 1.6732632423543772848170429916717
#scale =1.0507009873554804934193349852946


@op_register(OpType.SELU)
def selu(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]

    if self.quantized:
        x = inp.betensor
        x = x - inp.qmin
        lut = self.constants["lut"].betensor
        x = torch.reshape(x, (-1,))
        y = lookup_lut_powerof2(x, lut, inp.qbits, False, dtype2bits(
            self.constants["lut"].dtype), is_signed(self.constants["lut"].dtype))
        out.betensor = torch.reshape(y, inp.betensor.shape)
    else:
        alpha = self.get_param("alpha")
        gamma = self.get_param("gamma")
        out.betensor = gamma*torch.nn.functional.elu(inp.betensor, alpha)
        # out.betensor = torch.nn.functional.selu(inp.betensor)

    return out.betensor


@quant_register(OpType.SELU)
def selu_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    dev = inp.betensor.device
    q_mode_activation = self.attrs["q_mode_activation"]

    alpha = self.get_param("alpha")
    gamma = self.get_param("gamma")
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")

    q_bits_activation = self.attrs["q_bits_activation"]
    out.qinvariant = False
    out.qbits = q_bits_activation
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, True)

    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut = linear_dequantize(torch.linspace(inp.qmin, inp.qmax, steps=lsteps, device=dev), inp.scale, inp.zerop)
    lut = gamma*torch.nn.functional.elu(lut, alpha)
    lut = linear_quantize_clip(lut, out.scale, out.zerop, out.qmin, out.qmax)
    self.constants["lut"] = PyTensor(self.name+"/selu_lut", lut.cpu().numpy().astype(dtype2nptype(out.dtype)))

    self.params.pop('alpha')
    self.params.pop('gamma')
