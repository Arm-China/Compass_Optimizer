# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
import torch


def generate_pseudo_random_seq(input_length, output_length, seed):
    cum_seq = torch.zeros([output_length + 1])
    diff = torch.zeros([output_length])

    alpha = float(input_length) / output_length
    k = input_length // output_length
    u_max1 = (k + 2) / alpha - 1
    u_max2 = (input_length + 1 - k) / alpha - (output_length - 1)
    max_u = min(u_max1, u_max2)

    if seed != 0:
        mt = mersenne_twister_engine(mt19937_params)
        mt.set_seed(seed)
        rand_value = uniform_real_distribution(0, 1, mt).getrand()
    else:
        rand_value = torch.rand([1]).item()
    u = rand_value * max_u

    cum_seq[0] = 1
    cum_seq[output_length] = input_length + 1
    update_value = [alpha * (i + u) for i in range(1, output_length)]
    cum_seq[1:output_length] = torch.ceil(torch.tensor(update_value))

    for i in range(output_length):
        diff[i] = cum_seq[i + 1] - cum_seq[i]

    return diff


def generate_random_seq(input_length, output_length, seed):
    k = input_length // output_length
    num_random_spot = input_length % output_length
    diff = torch.ones([output_length]) * k
    diff[:num_random_spot] += 1
    if seed == 0:
        diff = shuffle(diff, None, 1, diff.shape[0])
    else:
        mt = mersenne_twister_engine(mt19937_params)
        mt.set_seed(seed)
        diff = shuffle(diff, mt, 0, diff.shape[0])
    return diff


def GeneratePoolingSequence(in_length, out_length, pseudo, seed):
    if pseudo:
        diff = generate_pseudo_random_seq(in_length, out_length, seed)
    else:
        diff = generate_random_seq(in_length, out_length, seed)
    cum_seq = torch.zeros([out_length + 1])
    for index in range(1, cum_seq.shape[0]):
        cum_seq[index] = cum_seq[index - 1] + diff[index - 1]
    return cum_seq.int()


@op_register(OpType.FractionalPool)
def fractional_pool2d(self, *args):
    _SUPPORTED_METHOD = ['MAX', 'AVG']
    method = self.get_param('method').upper()
    seed = int(self.get_param('seed'))
    pseudo = self.get_param('pseudo')
    overlap = self.get_param('overlap')
    if method not in _SUPPORTED_METHOD:
        (OPT_FATAL('layerid=%s, fractional pooling op only support %s, but now method=%s' %
                   (self.attrs['layer_id'], str(_SUPPORTED_METHOD), method)))

    inp = self.inputs[0].betensor
    batch = inp.shape[0]
    in_channel = self.inputs[0].ir_shape[3]
    in_height = self.inputs[0].ir_shape[1]
    in_width = self.inputs[0].ir_shape[2]
    out_height = self.outputs[0].ir_shape[1]
    out_width = self.outputs[0].ir_shape[2]

    out = torch.zeros([batch, out_height, out_width, in_channel], device=inp.device)

    height_cum_seq = GeneratePoolingSequence(in_height, out_height, pseudo, seed).to(inp.device)
    width_cum_seq = GeneratePoolingSequence(in_width, out_width, pseudo, seed).to(inp.device)

    StartH = height_cum_seq[:out_height]
    EndH = height_cum_seq[1: out_height+1] + (1 if overlap else 0)
    StartW = width_cum_seq[:out_width]
    EndW = width_cum_seq[1: out_width+1] + (1 if overlap else 0)
    EndH = torch.clamp(EndH, 0, in_height)
    EndW = torch.clamp(EndW, 0, in_width)

    for h_idx in range(out_height):
        for w_idx in range(out_width):
            inp_windows = inp[:, StartH[h_idx]:EndH[h_idx], StartW[w_idx]:EndW[w_idx], :]
            if method == "MAX":
                out[:, h_idx, w_idx, :] = torch.amax(inp_windows, dim=[1, 2])
            else:
                kernel_y = EndH[h_idx] - StartH[h_idx]
                kernel_x = EndW[w_idx] - StartW[w_idx]
                psum = torch.sum(inp_windows, dim=[1, 2])
                area = (kernel_x * kernel_y)
                if self.quantized:
                    do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
                        1. / area, mult_bits=8, force_shift_positive=False)
                    out[:, h_idx, w_idx, :] = linear_requantize(
                        psum, do_scale, do_shift, 0, self.outputs[0].qmin, self.outputs[0].qmax).int()
                else:
                    out[:, h_idx, w_idx, :] = psum / area

    self.outputs[0].betensor = out
    self.outputs[1].betensor = height_cum_seq
    self.outputs[2].betensor = width_cum_seq

    return (out, height_cum_seq, width_cum_seq)


@quant_register(OpType.FractionalPool)
def fractional_pool2d_quantize(self, *args):
    inp = self.inputs[0]
    out0 = self.outputs[0]

    out0.scale = inp.scale
    out0.zerop = inp.zerop
    out0.dtype = inp.dtype
    out0.qbits = inp.qbits
    out0.qmin = inp.qmin
    out0.qmax = inp.qmax
    out0.qinvariant = inp.qinvariant

    for i in range(1, len(self.outputs)):
        out = self.outputs[i]
        out.scale = 1
        out.zerop = 0
        act_bits, _ = range2bits(0, out.ir_shape[0], force_int=True)
        out.dtype = bits2dtype(max(16, act_bits), True)
        out.qbits = dtype2bits(out.dtype)
        out.qmin, out.qmax = dtype2range(out.dtype)
        out.qinvariant = True
