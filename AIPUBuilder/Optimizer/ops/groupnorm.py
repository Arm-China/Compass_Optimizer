# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.conv import linear_op_quantize
import torch

register_optype('GroupNorm')


@quant_register(OpType.GroupNorm)
def groupnorm_quantize(self, *args):
    q_mode_weight = self.attrs["q_mode_weight"]
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_weight = self.attrs["q_bits_weight"]
    q_bits_bias = self.attrs["q_bits_bias"]
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    ngamma = self.placeholders[0]
    normalized = self.placeholders[1]
    ngamma.qbits = q_bits_activation
    ngamma.scale, ngamma.zerop, ngamma.qmin, ngamma.qmax, ngamma.dtype = get_linear_quant_params_from_tensor(
        ngamma, q_mode_activation, ngamma.qbits, is_signed=False)
    ngamma.qinvariant = False
    normalized.qbits = q_bits_activation
    normalized.scale, normalized.zerop, normalized.qmin, normalized.qmax, normalized.dtype = get_linear_quant_params_from_tensor(
        normalized, q_mode_activation, normalized.qbits, is_signed=True)
    normalized.qinvariant = False

    ngamma_multiplier_bits = max(16, q_bits_activation)
    ngamma_do_scale, ngamma_do_scale_type, ngamma_do_shift, ngamma_do_shift_type = \
        get_scale_approximation_params(inp.scale * ngamma.scale,
                                       mult_bits=ngamma_multiplier_bits,
                                       force_shift_positive=self.force_shift_positive)
    self.params['ngamma_scale_value'] = int(ngamma_do_scale)
    self.params['ngamma_scale_type'] = ngamma_do_scale_type
    self.params['ngamma_shift_value'] = int(ngamma_do_shift)
    self.params['ngamma_shift_type'] = ngamma_do_shift_type
    self.params['ngamma_zp_value'] = int(ngamma.zerop)
    self.params['ngamma_zp_type'] = bits2dtype(q_bits_activation, is_signed=False)
    norm_do_scale, norm_do_scale_type, norm_do_shift, norm_do_shift_type = \
        get_scale_approximation_params(normalized.scale / (inp.scale * ngamma.scale),
                                       mult_bits=q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)
    self.params['norm_scale_value'] = int(norm_do_scale)
    self.params['norm_scale_type'] = norm_do_scale_type
    self.params['norm_shift_value'] = int(norm_do_shift)
    self.params['norm_shift_type'] = norm_do_shift_type

    w_scale = 1.0
    # if 'weights' in self.constants :
    #     w = self.constants['weights']
    #     w.scale, w.zerop, w.qmin, w.qmax = get_linear_quant_params_from_tensor(w, q_mode_weight, q_bits_weight, is_signed=True)
    #     w.betensor = linear_quantize_clip(w.betensor, w.scale, w.zerop, w.qmin, w.qmax)
    #     w.qbits = q_bits_weight
    #     w.dtype = bits2dtype(w.qbits, is_signed=True)
    #     w_scale = w.scale
    # if 'biases' in self.constants :
    #     b = self.constants["biases"]
    #     b.scale = normalized.scale * w_scale
    #     b.zerop = 0
    #     b.qmin = -2**(q_bits_bias-1)
    #     b.qmax = 2**(q_bits_bias-1) - 1
    #     b.qbits = q_bits_bias
    #     b.betensor = linear_quantize_clip(b.betensor, b.scale, b.zerop, b.qmin, b.qmax)
    #     b.dtype = bits2dtype(b.qbits, is_signed=True)
    if 'weights' in self.constants:
        old = self.replace_input_temporarily(0, normalized)
        uflag = self.attrs['unify_shifts_for_aiff']
        if self.type != OpType.GroupNorm:
            self.attrs['unify_shifts_for_aiff'] = False
        linear_op_quantize(self, *args)
        self.attrs['unify_shifts_for_aiff'] = uflag
        self.replace_input_temporarily(0, old)
    else:
        out = self.outputs[0]
        out.qbits = q_bits_activation
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, out.qbits, is_signed=True)
        out.qinvariant = False

        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(out.scale / (normalized.scale * w_scale),
                                           mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)
        self.params['scale_value'] = int(do_scale)
        self.params['scale_type'] = do_scale_type
        self.params['shift_value'] = int(do_shift)
        self.params['shift_type'] = do_shift_type

    self.constants["lut"] = PyTensor(
        self.name+"/isqrt_lut", torch.tensor(inverse_sqrt_table).cpu().numpy().astype(dtype2nptype(Dtype.INT16)))
    self.constants["lut"].dtype = Dtype.INT16


@op_register(OpType.GroupNorm)
def groupnorm(self, *args):
    axis = self.get_param('axis')
    eps = self.get_param('epsilon')
    group = self.get_param('group')
    inp = self.inputs[0]
    inp_betensor = inp.betensor.long() if self.quantized else inp.betensor.float()
    out = self.outputs[0]
    dim_num = inp_betensor.dim()
    channel_axis = dim_num - 1
    is_groupnorm = True if self.type == OpType.GroupNorm else False
    # groupnorm axis is channel axis
    if is_groupnorm:
        channel_axis = axis
        if channel_axis < 0:
            channel_axis += dim_num
        axis = list(range(1, dim_num))  # axis=[H, W, C//G], 0-dim is batch
        in_chunks = inp_betensor.split(inp_betensor.shape[channel_axis] // group, dim=channel_axis)
        axis_reshape = [inp_betensor.shape[channel_axis] if ax == channel_axis else 1 for ax in range(dim_num)]
        scale_shift_shape = axis_reshape
    # instancenorm and layernorm axis is normalization axis
    else:
        in_chunks = [inp_betensor]
        default_axis_reshape = [inp_betensor.shape[ax] if ax in axis else 1 for ax in range(dim_num)]
        axis_reshape = self.get_param('axis_shape', optional=True, default_value=default_axis_reshape)
        default_axis_reshape = [inp_betensor.shape[ax] if ax == axis[0] else 1 for ax in range(dim_num)]
        scale_shift_shape = self.get_param('scale_shift_shape', optional=True, default_value=default_axis_reshape)
    mean_chunks = []
    ngamma_chunks = []
    for t in in_chunks:
        if self.quantized:
            t_mean = torch.mean(t.double(), dim=axis, keepdim=True).round().long()
            t_var = torch.mean((t.double() - t_mean)**2, dim=axis, keepdim=True).long()

            # ng = isqrt(var) = isqrt(mean((inp - m)*(inp - m)))
            # (ng + z1)/s1 = isqrt(mean((inp - m)*(inp - m)/s0/s0))
            # (ng + z1)/s1 = isqrt(mean((inp - m)*(inp - m))) * s0
            # (ng + z1)/s1 = isqrt(var) * s0
            # (ng + z1) = isqrt(var) * (s0 * s1)
            ngamma_do_scale = self.params['ngamma_scale_value']
            ngamma_do_shift = self.params['ngamma_shift_value']
            ngamma_qmin, ngamma_qmax = bits2range(out.qbits, is_signed=False)
            # t_ngamma = (torch.pow(t_var, -0.5) * (2**31)).round()
            t_ngamma = calculate_inverse_sqrt(t_var)
            t_ngamma = t_ngamma >> 16
            t_ngamma = linear_requantize(t_ngamma, ngamma_do_scale, ngamma_do_shift + 15, 0, ngamma_qmin, ngamma_qmax)
            mean_chunks.append(t_mean)
            ngamma_chunks.append(t_ngamma)
        else:
            t_mean = torch.mean(t.float(), dim=axis, keepdim=True)
            t_var = torch.mean((t.float() - t_mean)**2, dim=axis, keepdim=True)
            # if t.shape[axis] == 1, t_std is nan using torch.std_mean
            #t_std, t_mean = torch.std_mean(t, dim=axis, keepdim=True)
            t_std = torch.pow(t_var + eps, 0.5)
            if torch.count_nonzero(t_std) != t_std.numel():
                OPT_WARN('type=%s, input std contain zero value, please check the axis or set epsilon nonzero value' % (str(self.type)))
            t_ngamma = 1.0 / (t_std)
            mean_chunks.append(t_mean)
            ngamma_chunks.append(t_ngamma)
    mean = torch.cat(mean_chunks, dim=channel_axis)
    ngamma = torch.cat(ngamma_chunks, dim=channel_axis)
    if is_groupnorm and mean.shape[channel_axis] != inp_betensor.shape[channel_axis]:
        mean = torch.repeat_interleave(mean, inp_betensor.shape[channel_axis]//group, dim=channel_axis)
        ngamma = torch.repeat_interleave(ngamma, inp_betensor.shape[channel_axis]//group, dim=channel_axis)
    normalized = (inp_betensor - mean) * ngamma
    if self.quantized:
        qmin, qmax = bits2range(out.qbits, is_signed=True)
        normalized = linear_requantize(normalized,
                                       self.params['norm_scale_value'], self.params['norm_shift_value'], 0,
                                       qmin, qmax)
    out.betensor = normalized
    gamma = 1.0
    if 'weights' in self.constants:
        w = self.constants["weights"]
        # must be positive axis
        w.key_axis_c = len(w.shape)-1
        w_zerop = w.zerop
        if isinstance(w.zerop, torch.Tensor):
            zerop_shape = [w.shape[ax] if ax == w.key_axis_c else 1 for ax in range(len(w.shape))]
            w_zerop = torch.reshape(w.zerop, zerop_shape)
        gamma = w.betensor + w_zerop
        gamma = torch.reshape(gamma, axis_reshape)
    beta = 0.0
    if 'biases' in self.constants:
        b = self.constants["biases"]
        b.key_axis_c = len(b.shape)-1
        beta = b.betensor + b.zerop
        beta = torch.reshape(beta, axis_reshape)
    out.betensor = gamma * normalized + beta
    if self.quantized:
        if "shift" not in self.constants:
            do_shift = self.params["shift_value"]
            do_scale = self.params["scale_value"]
        else:
            do_shift = self.constants["shift"].betensor
            do_scale = self.constants["scale"].betensor
            do_shift = torch.reshape(do_shift, scale_shift_shape)
            do_scale = torch.reshape(do_scale, scale_shift_shape)
        o_zp = out.zerop  # if 'biases' not in self.constants else 0 # biases has absorbed out.zerop
        out.betensor = linear_requantize(out.betensor,
                                         do_scale, do_shift, o_zp,
                                         out.qmin, out.qmax)
    if not self.quantized:
        if len(self.placeholders) < 1:
            ph0 = PyTensor(self.name+"/ngamma", ngamma.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            ph1 = PyTensor(self.name+"/normalized", normalized.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            self.placeholders.append(ph0)
            self.placeholders.append(ph1)
        self.placeholders[0].betensor = ngamma
        self.placeholders[1].betensor = normalized

    return out.betensor
