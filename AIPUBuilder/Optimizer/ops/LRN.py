# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.utils import get_linear_quant_params_from_tensor as get_qinfo_from_tensor
from AIPUBuilder.Optimizer.logger import *
import torch


def _get_lut_bits(q_activation_bits, offset_bits):
    lut_bits = (q_activation_bits + offset_bits) * 2
    return lut_bits


@op_register(OpType.LRN)
def LRN(self, *args):
    _SUPPORT_METHOD = ['ACROSS_CHANNELS', 'WITHIN_CHANNEL']
    inpt = self.inputs[0].betensor
    method = self.get_param('method').upper()
    if method not in _SUPPORT_METHOD:
        OPT_ERROR("id=%s,op_type=%s DoesNot support '%s', only support %s"
                  % (self.attrs['layer_id'], str(self.type), method, str(_SUPPORT_METHOD)))

    size = self.get_param('size')
    bias = self.get_param('bias')
    alpha = self.get_param('alpha')
    beta = self.get_param('beta')
    input = nhwc2nchw(inpt)
    if not self.quantized:
        inputs_square = torch.square(input)
        if method == 'ACROSS_CHANNELS':
            # outt = torch.nn.functional.local_response_norm(input, size, alpha=alpha, beta=beta, k=bias)
            inputs_square = inputs_square.unsqueeze(1)
            padding = (0, 0, 0, 0, int(size / 2), int((size - 1) / 2))
            inp_square_with_pad = torch.nn.functional.pad(inputs_square, padding, mode='constant', value=0)
            square_sum = torch.nn.functional.avg_pool3d(inp_square_with_pad,
                                                        kernel_size=(size, 1, 1),
                                                        stride=1,
                                                        padding=0,
                                                        divisor_override=1,
                                                        count_include_pad=False)
            square_sum = square_sum.squeeze(1) / size
            denominator = torch.pow((bias + alpha * square_sum), beta)
            outt = input / denominator
            if len(self.placeholders) < 1:
                ph0 = PyTensor(self.name+"/square_sum", square_sum.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
                self.placeholders.append(ph0)
            self.placeholders[0].betensor = square_sum
        else:  # method == 'WITHIN_CHANNEL':
            square_sum = torch.nn.functional.avg_pool2d(inputs_square,
                                                        kernel_size=size,
                                                        stride=1,
                                                        padding=int((size-1.0)/2),
                                                        # divisor_override=1,
                                                        count_include_pad=False)
            denominator = torch.pow((bias + alpha * square_sum), beta)
            outt = input / denominator
    else:
        input = input + self.inputs[0].zerop
        do_scale, do_scale_n = self.get_param('scale_value'), self.get_param('scale_sum_value')
        do_shift, do_shift_n = self.get_param('shift_value'), self.get_param('shift_sum_value')
        lut = self.constants['lut']
        # already requantize the square_sum 18bits to 16bits using do_scale_n and do_shift_n
        # offset_bits = 0
        # lut_in_bits = _get_lut_bits(self.inputs[0].qbits, offset_bits)
        lut_in_bits = 16
        lut_in_qmin, lut_in_qmax = bits2range(lut_in_bits, False)
        lut_out_bits = 16

        inputs_square = torch.square(input)
        dtype = inputs_square.type()
        if method == 'ACROSS_CHANNELS':
            inputs_square = inputs_square.unsqueeze(1)
            padding = (0, 0, 0, 0, int(size / 2), int((size - 1) / 2))
            inp_square_with_pad = torch.nn.functional.pad(inputs_square, padding, mode='constant', value=0)
            square_sum = torch.nn.functional.avg_pool3d(inp_square_with_pad.type(torch.float32),
                                                        kernel_size=(size, 1, 1),
                                                        stride=1,
                                                        padding=0,
                                                        divisor_override=1,
                                                        count_include_pad=False)
        else:  # method == 'WITHIN_CHANNEL'
            square_sum = torch.nn.functional.avg_pool2d(inputs_square.type(torch.float32),
                                                        kernel_size=size,
                                                        stride=1,
                                                        padding=int((size-1.0)/2),
                                                        divisor_override=1,
                                                        count_include_pad=False)

        avg_pool = linear_requantize(square_sum.type(dtype), do_scale_n, do_shift_n, 0, lut_in_qmin, lut_in_qmax)
        avg_pool = torch.reshape(avg_pool.squeeze(1), (-1,))
        in_is_signed = False  # becasue the square_sum is unsigned.
        lut_v = lookup_lut_powerof2(avg_pool,
                                    lut.betensor,
                                    lut_in_bits,
                                    in_is_signed,
                                    lut_out_bits,
                                    is_signed(lut.dtype))
        lut_v = torch.reshape(lut_v, input.shape)
        outt = input * lut_v
        outt = linear_requantize(outt,
                                 do_scale,
                                 do_shift,
                                 self.outputs[0].zerop,
                                 self.outputs[0].qmin,
                                 self.outputs[0].qmax)

    outt = nchw2nhwc(outt)
    self.outputs[0].betensor = outt
    return outt


@quant_register(OpType.LRN)
def LRN_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    size = self.get_param('size')
    bias = self.get_param('bias')
    alpha = self.get_param('alpha')
    beta = self.get_param('beta')

    q_bits_activation = self.attrs['q_bits_activation']
    q_mode_activation = self.attrs['q_mode_activation']
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_ERROR("Currently not support per-channel quantization of activations, Optmizer will use per-layer quantization for LRN OP")
    out_signed = is_signed(inp.dtype)
    out.qbits = q_bits_activation
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_qinfo_from_tensor(out,
                                                                                q_mode_activation,
                                                                                out.qbits,
                                                                                is_signed=out_signed)
    out.qinvariant = False

    placeholder_scale = 1.
    if len(self.placeholders) > 0:
        placeholders = self.placeholders[0]
        placeholders_qinfo = get_qinfo_from_tensor(placeholders,
                                                   QuantMode.to_symmetric(q_mode_activation),
                                                   16,
                                                   is_signed=False)
        placeholder_scale = placeholders_qinfo[0]

    # make the lut
    # offset_bits = 1 if QuantMode.is_asymmetric(q_mode_activation) else 0
    # offset_bits = 1 if self.inputs[0].zerop else 0
    offset_bits = 0
    # lut_in_bits = _get_lut_bits(self.inputs[0].qbits, offset_bits)
    lut_in_bits = 16
    lut_qmin, lut_qmax = bits2range(lut_in_bits, False)
    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut = 1. / torch.pow(bias + alpha * (torch.linspace(lut_qmin, lut_qmax, steps=lsteps) / placeholder_scale), beta)
    lut = PyTensor(self.name+"/tmp", lut.cpu().numpy())
    lut.min = lut.betensor.min().item()
    lut.max = lut.betensor.max().item()
    # lut output bits fixed to 16bits
    # lut_out_bits = 16
    lut_out_bits = self.attrs['scaling_bits'][0]
    lut.qbits = lut_out_bits
    lut.scale, lut.zerop, lut.qmin, lut.qmax, lut.dtype = get_qinfo_from_tensor(lut,
                                                                                QuantMode.to_symmetric(
                                                                                    q_mode_activation),
                                                                                lut.qbits,
                                                                                is_signed=False)
    qlut = linear_quantize_clip(lut.betensor, lut.scale, lut.zerop, lut.qmin, lut.qmax)
    self.constants['lut'] = PyTensor(self.name+"/lrn_lut", qlut.cpu().numpy().astype(dtype2nptype(lut.dtype)))

    # quantize the 1./ size
    total_rescale = placeholder_scale / (inp.scale * inp.scale * size)
    scl_n, scl_ntype, sft_n, sft_ntype = get_scale_approximation_params(total_rescale,
                                                                        mult_bits=q_bits_activation,
                                                                        force_shift_positive=self.force_shift_positive)

    total_scale = out.scale / inp.scale / lut.scale
    dscale, dscale_t, dshift, dshift_t = get_scale_approximation_params(total_scale,
                                                                        mult_bits=q_bits_activation,
                                                                        force_shift_positive=self.force_shift_positive)
    self.params['scale_sum_value'] = int(scl_n)
    self.params['shift_sum_value'] = int(sft_n)
    self.params['scale_sum_type'] = scl_ntype
    self.params['shift_sum_type'] = sft_ntype
    self.params['scale_value'] = int(dscale)
    self.params['shift_value'] = int(dshift)
    self.params['scale_type'] = dscale_t
    self.params['shift_type'] = dshift_t
