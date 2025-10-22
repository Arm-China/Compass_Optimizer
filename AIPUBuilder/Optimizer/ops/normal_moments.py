# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.ops.eltwise import calc_eltwise_add_like_scale_shift

import torch

register_optype('NormalizedMoments')


@quant_register(OpType.NormalizedMoments)
def normalizedMoments_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]
    multiplier_bits = self.attrs['multiplier_bits']
    count = int(self.get_param("counts"))

    mean_t = self.inputs[0]
    var_t = self.inputs[1]
    shift_t = self.inputs[2]
    out0 = self.outputs[0]
    out1 = self.outputs[1]
    plh = self.placeholders[0]

    out0.qbits = q_bits_activation
    out0.qinvariant = False
    out0_sign = (is_signed(mean_t.dtype) and is_signed(shift_t.dtype)) or self.force_dtype_int
    out0.scale, out0.zerop, out0.qmin, out0.qmax, out0.dtype = get_linear_quant_params_from_tensor(
        out0, q_mode_activation, out0.qbits, out0_sign)

    out1.qbits = q_bits_activation
    out1.qinvariant = False
    out1_sign = True
    out1.scale, out1.zerop, out1.qmin, out1.qmax, out1.dtype = get_linear_quant_params_from_tensor(
        out1, q_mode_activation, out1.qbits, out1_sign)

    plh.qbits = q_bits_activation
    plh.qinvariant = False
    plh.scale, plh.zerop, plh.qmin, plh.qmax, plh.dtype = get_linear_quant_params_from_tensor(
        plh, QuantMode.to_symmetric(q_mode_activation), plh.qbits, False)

    # unify mean + shift scale
    _, clip_max = dtype2range(Dtype.INT16)
    mean_clip_max = var_clip_max = clip_max
    mean_placeholder = PyTensor(self.name + '/mean_placeholder')
    mean_placeholder.scale = mean_t.scale * count
    if (mean_t.zerop != 0) or (shift_t.zerop != 0):
        mean_clip_max //= 2
    mean_scale0, mean_scale1, mean_do_scale, mean_do_shift, mean_do_scale_type, mean_do_shift_type, _ = calc_eltwise_add_like_scale_shift(mean_placeholder,
                                                                                                                                          shift_t,
                                                                                                                                          out0,
                                                                                                                                          mean_clip_max,
                                                                                                                                          multiplier_bits,
                                                                                                                                          self.type,
                                                                                                                                          self.attrs['layer_id'])

    # generate lut for mean*mean
    lsteps = 2 ** min(mean_t.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut = linear_dequantize(torch.linspace(mean_t.qmin, mean_t.qmax, steps=lsteps), mean_t.scale, mean_t.zerop)
    lut = torch.square(lut/count)
    lut = linear_quantize_clip(lut, plh.scale, plh.zerop, plh.qmin, plh.qmax)
    self.constants['lut'] = PyTensor(
        self.name+'/lut', lut, dtype=bits2dtype(q_bits_activation, False))
    self.constants['lut'].dtype = bits2dtype(q_bits_activation, is_signed=False)

    # unify var - mean*mean scale
    var_placeholder = PyTensor(self.name + '/var_placeholder')
    var_placeholder.scale = var_t.scale * count
    if var_t.zerop != 0:
        var_clip_max = var_clip_max // 2
    var_scale0, var_scale1, var_do_scale, var_do_shift, var_do_scale_type, var_do_shift_type, _ = calc_eltwise_add_like_scale_shift(var_placeholder,
                                                                                                                                    plh,
                                                                                                                                    out1,
                                                                                                                                    var_clip_max,
                                                                                                                                    multiplier_bits,
                                                                                                                                    self.type,
                                                                                                                                    self.attrs['layer_id'])

    self.params["mean_shift_value"] = int(mean_do_shift)
    self.params["mean_shift_type"] = mean_do_shift_type
    self.params["mean_scale_value"] = [int(mean_do_scale), int(mean_scale0), int(mean_scale1)]
    self.params["mean_scale_type"] = [Dtype.UINT16, Dtype.UINT16, Dtype.UINT16]

    self.params["var_shift_value"] = int(var_do_shift)
    self.params["var_shift_type"] = var_do_shift_type
    self.params["var_scale_value"] = [int(var_do_scale), int(var_scale0), int(var_scale1)]
    self.params["var_scale_type"] = [Dtype.UINT16, Dtype.UINT16, Dtype.UINT16]


@op_register(OpType.NormalizedMoments)
def normalizedMoments(self, *args):
    mean_t = self.inputs[0]
    var_t = self.inputs[1]
    shift_t = self.inputs[2]
    out0 = self.outputs[0]
    out1 = self.outputs[1]
    counts = int(self.get_param("counts"))

    if self.quantized:
        mean_scale, mean_scale0, mean_scale1 = self.params["mean_scale_value"]
        var_scale, var_scale0, var_scale1 = self.params["var_scale_value"]
        mean_shift = self.params["mean_shift_value"]
        var_shift = self.params["var_shift_value"]
        lut = self.constants['lut']

        #mean_out = mean/count + shift
        scaled_mean = (mean_t.betensor.long() + mean_t.zerop) * mean_scale0
        scaled_shift = (shift_t.betensor.long() + shift_t.zerop) * mean_scale1
        shifted_mean = scaled_mean + scaled_shift
        mean_out = linear_requantize(shifted_mean, mean_scale, mean_shift, out0.zerop, out0.qmin, out0.qmax)

        #var_out = var/count - (mean/count)^2
        mean_flatten = torch.reshape(mean_t.betensor, (-1,))
        mean_flatten = lookup_lut_powerof2(mean_flatten, lut.betensor, mean_t.qbits,
                                           is_signed(mean_t.dtype), dtype2bits(lut.dtype), False)
        square_mean = torch.reshape(mean_flatten, mean_t.betensor.shape)
        scaled_var = (var_t.betensor.long() + var_t.zerop) * var_scale0
        square_mean = square_mean * var_scale1
        var_out = scaled_var - square_mean
        var_out = linear_requantize(var_out, var_scale, var_shift, out1.zerop, out1.qmin, out1.qmax)

    else:
        shifted_mean = mean_t.betensor.float() / counts
        mean_out = shifted_mean + shift_t.betensor.float()
        mean_square = shifted_mean * shifted_mean
        var_out = var_t.betensor.float() / counts - mean_square
        if len(self.placeholders) < 1:
            ph0 = PyTensor(self.name+"/mean_square",
                           mean_square, dtype=Dtype.FP32)
            self.placeholders.append(ph0)
        self.placeholders[0].betensor = mean_square

    out0.betensor = mean_out
    out1.betensor = var_out
    return (out0.betensor, out1.betensor)
