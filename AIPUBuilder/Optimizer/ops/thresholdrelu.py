# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch

# y = x for x > alpha, y = 0 otherwise, is applied to the tensor elementwise.

register_optype('THRESHOLDEDRELU')


@quant_register(OpType.THRESHOLDEDRELU)
def thresholdedrelu_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    out = self.outputs[0]

    alpha = float(self.get_param("alpha"))

    out.qbits = q_bits_activation
    out_sign = True if alpha < 0.0 else False
    dev = inp.betensor.device
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, out_sign or self.force_dtype_int)
    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut = linear_dequantize(torch.linspace(inp.qmin, inp.qmax, steps=lsteps, device=dev), inp.scale, inp.zerop)
    lut = torch.nn.functional.threshold(lut, alpha, 0)
    lut = linear_quantize_clip(lut, out.scale, out.zerop, out.qmin, out.qmax)
    self.constants["lut"] = PyTensor(self.name+"/thresholdedrelu_lut",
                                     lut.cpu().numpy().astype(dtype2nptype(out.dtype)))
    out.qinvariant = False

    self.params.pop('alpha')


@op_register(OpType.THRESHOLDEDRELU)
def thresholdedrelu(self, *args):
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
        alpha = float(self.get_param("alpha"))
        out.betensor = torch.nn.functional.threshold(inp.betensor, alpha, 0)

    return out.betensor


def threshold_out_signed(self):
    alpha = float(self.get_param("alpha"))
    return False if alpha >= 0 else True
