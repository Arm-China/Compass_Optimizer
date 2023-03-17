# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import OPT_FATAL
from AIPUBuilder.Optimizer.utils import *
import torch


@op_register(OpType.Gemm)
def gemm(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]
    x = inp0.betensor
    y = inp1.betensor
    if self.get_param('trans_a'):
        if x.dim() == 0:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.transpose(-1, -2)
    if self.get_param('trans_b'):
        if y.dim() == 0:
            y = y.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 1:
            y = y.unsqueeze(0)
        y = y.transpose(-1, -2)
    if self.quantized:
        x = x + inp0.zerop
        y = y + inp1.zerop
    z = torch.matmul(x, y)
    alpha = self.get_param('alpha')
    if self.quantized:
        do_scale = self.params["scale_value"]
        do_shift = self.params['shift_value']
        alph_sign = -1 if alpha < 0 else 1
        if len(self.inputs) == 3:
            '''do_scale=[doscale, inp_scale0, inp_scale1]'''
            beta = self.get_param('beta')
            beta_sign = -1 if beta < 0 else 1
            bias = self.inputs[2].betensor + self.inputs[2].zerop
            req_z = torch.round(z * do_scale[1]) * alph_sign
            req_bias = torch.round(bias * do_scale[2]) * beta_sign
            sum = req_z + req_bias
            z = linear_requantize(sum, do_scale[0], do_shift, out.zerop, out.qmin, out.qmax)
        else:  # len(self.inputs) == 2
            z = linear_requantize(z, do_scale, do_shift, out.zerop, out.qmin, out.qmax)
            z = z * alph_sign
    else:
        z = alpha * z
        if len(self.inputs) == 3:
            beta = self.get_param('beta')
            inp2 = self.inputs[2]
            z = z + inp2.betensor * beta
    out.betensor = z
    return out.betensor


@quant_register(OpType.Gemm)
def gemm_quantize(self, *args):
    '''
    if C is existed:
        y = alpha * A' * B' + beta * C
        yq / ys = alpha * Aq / As * Bq / Bs + beta * Cq / Cs
        yq = ys(alpha/(As * Bs) * (Aq * Bq) + beta / Cs * Cq)
        yq = ys/inp_max_scale * (inp_max_scale / (alpha/(As*Bs)) *(Aq*Bq) + inp_max_scale/(beta/Cs) *Cq)
        as:
        ts0 = ys / inp_max_scale == > S * 2 ** n
        s0 = inp_max_scale / (alpha/(As*Bs))
        s1 = inp_max_scale / (beta/Cs)
    else:
        matmul quantization:
        y = alpha * A' * B'
        yq / ys = alpha * Aq / As * Bq / Bs
        yq = alpha * ys / (As * Bs) * (Aq * Bq)
        as:
        alpha*ys/(As*Bs) ==> S * 2 ** n
    :param self:
    :param args:
    :return:
    '''
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_activation = self.attrs["q_bits_activation"]
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    alpha = self.get_param('alpha')
    beta = 1.0

    out = self.outputs[0]
    out_signed = is_signed(inp0.dtype) or is_signed(inp1.dtype) or alpha < 0
    out.qinvariant = inp0.qinvariant and inp1.qinvariant

    if len(self.inputs) == 3:
        inp2 = self.inputs[2]
        beta = self.get_param('beta')
        out_signed = out_signed or is_signed(inp2.dtype) or beta < 0
        out.qinvariant = out.qinvariant and inp2.qinvariant
    out_signed = out_signed or self.force_dtype_int

    if out.qinvariant:
        out.scale = 1.0
        out.zerop = 0
        out.qbits, out.dtype = range2dtype(out.extrema_min, out.extrema_max, force_int=self.force_dtype_int)
    else:
        out.qbits = q_bits_activation
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, out.qbits, is_signed=out_signed)

    if len(self.inputs) == 3:
        inp2 = self.inputs[2]

        branch0_scale = abs(alpha) / (inp0.scale * inp1.scale)
        branch1_scale = abs(beta) / inp2.scale
        inp_max_scale = max(branch0_scale, branch1_scale)
        q_branch0_scale = int((inp_max_scale / branch0_scale) * 256)
        q_branch1_scale = int((inp_max_scale / branch1_scale) * 256)

        total_scale = out.scale / inp_max_scale
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(total_scale,
                                           q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)
        self.params["shift_value"] = int(do_shift) + 8
        self.params["shift_type"] = do_shift_type
        self.params["scale_value"] = [int(do_scale), q_branch0_scale, q_branch1_scale]
        self.params["scale_type"] = [do_scale_type, Dtype.UINT16, Dtype.UINT16]

    elif len(self.inputs) == 2:
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(abs(alpha) * out.scale / (inp0.scale * inp1.scale),
                                           q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)
        self.params["shift_value"] = do_shift
        self.params["shift_type"] = do_shift_type
        self.params["scale_value"] = do_scale
        self.params["scale_type"] = do_scale_type
    else:
        OPT_ERROR(f"Gemm Op now support 2 or 3 inputs, but now input'num = {len(self.inputs)}.")
