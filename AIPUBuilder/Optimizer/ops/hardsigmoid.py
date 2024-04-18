# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch

register_optype('HardSigmoid')


@quant_register(OpType.HardSigmoid)
def hardsigmoid_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    out = self.outputs[0]
    alpha = self.get_param('alpha')
    beta = self.get_param('beta')
    # it will set optional to False when parser add 'clip_max' and 'clip_min' in future
    clip_max = self.get_param('clip_max', optional=True, default_value=1)
    clip_min = self.get_param('clip_min', optional=True, default_value=0)

    out.qbits = q_bits_activation
    out_sign = clip_min < 0 or self.force_dtype_int
    dev = inp.betensor.device
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, out_sign)
    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut = linear_dequantize(torch.linspace(inp.qmin, inp.qmax, steps=lsteps, device=dev), inp.scale, inp.zerop)
    hard_sigmoid_opt_tag = self.get_attrs('hard_sigmoid_opt_tag', optional=True, default_value=0)
    if hard_sigmoid_opt_tag != 0:
        lut = (lut - 1. / (2 * alpha))
    lut = torch.maximum(torch.tensor(clip_min, device=dev),
                        torch.minimum(alpha * lut + beta, torch.tensor(clip_max, device=dev)))
    # lut = torch.nn.functional.hardsigmoid(lut)
    lut = linear_quantize_clip(lut, out.scale, out.zerop, out.qmin, out.qmax)
    self.constants["lut"] = PyTensor(self.name+"/sigmoid_lut", lut.cpu().numpy().astype(dtype2nptype(out.dtype)))
    out.qinvariant = False


@op_register(OpType.HardSigmoid)
def hardsigmoid(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    alpha = self.get_param('alpha')
    beta = self.get_param('beta')
    hard_sigmoid_opt_tag = self.get_attrs('hard_sigmoid_opt_tag', optional=True, default_value=0)
    # it will set optional to False when parser add 'clip_max' and 'clip_min' in future
    clip_max = self.get_param('clip_max', optional=True, default_value=1)
    clip_min = self.get_param('clip_min', optional=True, default_value=0)

    x = inp.betensor
    dev = x.device
    if self.quantized:
        x = x - inp.qmin
        lut = self.constants["lut"].betensor
        x = torch.reshape(x, (-1,))
        # y = torch.gather(lut, 0, x.long())
        y = lookup_lut_powerof2(x, lut, inp.qbits, False, dtype2bits(
            self.constants["lut"].dtype), is_signed(self.constants["lut"].dtype))
        out.betensor = torch.reshape(y, inp.betensor.shape)
    else:
        if hard_sigmoid_opt_tag != 0:
            x = (x - 1. / (2 * alpha))
        out.betensor = torch.maximum(torch.tensor(clip_min, device=dev), torch.minimum(
            alpha * x + beta, torch.tensor(clip_max, device=dev)))
        # out.betensor = torch.nn.functional.hardsigmoid(x)
    return out.betensor


def hardsigmoid_out_signed(self):
    clip_min = self.get_param('clip_min', optional=True, default_value=0)
    return False if clip_min >= 0 else True
