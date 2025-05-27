# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch

# calculate 1/sqrt(x)


def isqrt_forward(x):
    xq = x.to(torch.int64)
    overflow_bits = torch.zeros_like(x).int()
    for k in range(31, 64):
        overflow_bits = torch.where((xq >> k) == 0, overflow_bits, overflow_bits + 1)
    overflow_bits = torch.where((overflow_bits % 2).to(torch.bool), overflow_bits + 1, overflow_bits)
    yq = calculate_inverse_sqrt(xq >> overflow_bits).to(torch.int64)
    rsqnorm = yq.long()
    rsqnorm_bits = torch.ceil(torch.log2(rsqnorm)).int()
    rsqnorm_limit_bits = 15
    rsqnorm_shift = torch.where(rsqnorm_bits > rsqnorm_limit_bits, rsqnorm_bits -
                                rsqnorm_limit_bits, torch.zeros_like(rsqnorm_bits, device=rsqnorm_bits.device).int())
    rsqnorm = torch.clamp(rsqnorm >> rsqnorm_shift, 0, 2**rsqnorm_limit_bits-1)
    return rsqnorm, rsqnorm_shift - torch.div(overflow_bits, 2, rounding_mode='trunc')


@quant_register(OpType.Normalization)
def lp_normalization_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_activation = self.attrs["q_bits_activation"]
    inp = self.inputs[0]
    out = self.outputs[0]
    out_sign = (inp.qmin + inp.zerop) < 0
    out.qinvariant = False
    out.qbits = q_bits_activation
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, is_signed=out_sign)
    do_scale, do_scale_type, do_shift, do_shift_type = \
        get_scale_approximation_params(out.scale, mult_bits=q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)
    self.params['scale_value'] = int(do_scale)
    self.params['scale_type'] = do_scale_type
    self.params['shift_value'] = int(do_shift)
    self.params['shift_type'] = do_shift_type
    method = self.get_param('method').lower()
    if method == 'l2':
        pre_shift = 0
        scale_bits = torch.ceil(torch.log2(do_scale)).item()
        input_bits, _ = range2bits(inp.qmin+inp.zerop, inp.qmax+inp.zerop, force_int=True)
        # always limit input*scale to int16
        if (input_bits + scale_bits) > 16:
            pre_shift = (input_bits + scale_bits) - 16
        self.params['reciprocal_shift_value'] = int(pre_shift)
        self.params['reciprocal_shift_type'] = Dtype.INT8
        self.constants["lut"] = PyTensor(
            self.name+"/isqrt_lut", torch.tensor(inverse_sqrt_table).cpu().numpy().astype(dtype2nptype(Dtype.INT16)))
        self.constants["lut"].dtype = Dtype.INT16


@op_register(OpType.Normalization)
def lp_normalization_forward(self, *args):
    axis = self.get_param('axis')
    # The order of the normalization, only 1 or 2 are supported.
    method = self.get_param('method').lower()
    eps = self.get_param('epsilon', optional=True, default_value=1e-12)
    eps = eps if eps != 0.0 else OPT_EPSILON
    p = 2
    if 'l1' == method:
        p = 1
    elif 'l2' == method:
        p = 2
    else:
        OPT_FATAL('%s: only support l1 and l2 normalization.' % (self.type,))
    inp = self.inputs[0]
    out = self.outputs[0]
    if self.quantized:
        x = inp.betensor.long() + inp.zerop
        psum = torch.sum(torch.abs(x)**p, dim=axis, keepdim=True).long()
        do_scale = self.get_param('scale_value')
        do_shift = self.get_param('shift_value')
        if 2 == p:
            q16_min, q16_max = bits2range(16, True)
            act_qmin, act_qmax = bits2range(32, True)
            pre_shift = self.get_param('reciprocal_shift_value')

            rsqnorm, rsqnorm_shift = isqrt_forward(psum)
            x = torch.clamp((x * do_scale), act_qmin, act_qmax)
            x = torch.clamp(x.long() >> pre_shift, q16_min, q16_max).to(x.dtype)
            x = torch.clamp((x * rsqnorm), act_qmin, act_qmax)
            out.betensor = linear_requantize(x, 1, do_shift + 31 - pre_shift -
                                             rsqnorm_shift, out.zerop, out.qmin, out.qmax)
        elif 1 == p:
            repeat_size = [x.shape[ax] if ax in axis else 1 for ax in range(psum.dim())]
            psum = psum.repeat(repeat_size)
            zeros_tensor = torch.zeros_like(x, device=psum.device)
            if do_shift < 0:
                psum = psum >> (0-do_shift)
                do_shift = 0
            x = torch.where(psum != 0, torch.trunc(x * do_scale / psum).long(), zeros_tensor.long())
            out.betensor = linear_requantize(x, 1, do_shift, out.zerop, out.qmin, out.qmax)

    else:
        x = inp.betensor.float()
        norm = torch.sum(torch.abs(x)**p, dim=axis, keepdim=True).float()
        norm[norm < eps] = eps
        norm = torch.pow(norm, 1/p)
        out.betensor = x / norm

    return out.betensor
