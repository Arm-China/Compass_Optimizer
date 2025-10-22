# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch


@quant_register(OpType.Div)
def div_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_activation = self.attrs["q_bits_activation"]
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]
    out_sign = is_signed(inp0.dtype) or is_signed(inp1.dtype)
    out.qinvariant = False
    out.qbits = q_bits_activation
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, is_signed=out_sign)

    lut_in_bits = inp1.qbits
    lut_in_qmin, lut_in_qmax = bits2range(lut_in_bits, is_signed=False)
    lut_zoom_value = 2 ** 16 - 1.
    lsteps = 2 ** min(inp1.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut = (lut_zoom_value / torch.linspace(lut_in_qmin, lut_in_qmax, steps=lsteps)).round()
    lut = torch.clamp(lut, 0, lut_zoom_value)
    # Dividing by 0 is an illegal operation and set it to 0 currently
    lut[0] = 0
    self.constants['lut'] = PyTensor(self.name+"/lut", lut, dtype=Dtype.UINT16)

    do_scale, do_scale_type, do_shift, do_shift_type = \
        get_scale_approximation_params((out.scale * inp1.scale) / (inp0.scale * lut_zoom_value),
                                       mult_bits=q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)
    self.params['scale_value'] = int(do_scale)
    self.params['scale_type'] = do_scale_type
    self.params['shift_value'] = int(do_shift)
    self.params['shift_type'] = do_shift_type


@op_register(OpType.Div)
def div_forward(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]
    if self.quantized:
        x = inp0.betensor + inp0.zerop
        y = inp1.betensor + inp1.zerop
        sign_y = torch.sign(y)
        # As with lib, the result of division by 0 is determined by lut[0]
        sign_y[sign_y == 0] = 1
        lut = self.constants['lut']
        lut_in_bits = inp1.qbits
        lut_v = lookup_lut_powerof2(torch.abs(y).reshape(-1), lut.betensor, lut_in_bits, False, 16, False)
        lut_v = torch.reshape(lut_v, y.shape)
        z = sign_y * x * lut_v
        do_scale = self.get_param('scale_value')
        do_shift = self.get_param('shift_value')
        out.betensor = linear_requantize(z, do_scale, do_shift, out.zerop, out.qmin, out.qmax)
    else:
        out.betensor = inp0.betensor / inp1.betensor
    return out.betensor
