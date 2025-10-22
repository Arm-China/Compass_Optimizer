# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.ops.conv import linear_op_quantize
from AIPUBuilder.Optimizer.utils.dtype_utils import construct_torch_tensor as torch_tensor
import torch
import math
register_optype('RMSNorm')


@quant_register(OpType.RMSNorm)
def RMSNorm_quantize(self, *args):
    q_mode_weight = self.attrs["q_mode_weight"]
    q_mode_activation = self.attrs["q_mode_activation"]
    q_bits_weight = self.attrs["q_bits_weight"]
    q_bits_bias = self.attrs["q_bits_bias"]
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]

    normalized = PyTensor(self.name+"/normalized")
    normalized.qbits = q_bits_activation
    # get need norm length N
    axis = self.get_param('axis')
    dim_num = len(inp.ir_shape)

    scope_norm = [inp.ir_shape[ax] if ax in axis else 1 for ax in range(dim_num)]

    N = math.prod(scope_norm)
    sqrt_n = math.sqrt(N)
    normalized.min = -sqrt_n
    normalized.max = sqrt_n
    normalized.scale, normalized.zerop, normalized.qmin, normalized.qmax, normalized.dtype = get_linear_quant_params_from_tensor(
        normalized, q_mode_activation, normalized.qbits, is_signed=True)
    normalized.qinvariant = False

    norm_do_scale, norm_do_scale_type, norm_do_shift, norm_do_shift_type = \
        get_scale_approximation_params(normalized.scale / 127,
                                       mult_bits=q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)

    self.params['norm_scale_value'] = int(norm_do_scale)
    self.params['norm_scale_type'] = norm_do_scale_type
    self.params['norm_shift_value'] = int(norm_do_shift)
    self.params['norm_shift_type'] = norm_do_shift_type

    w_scale = 1.0
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
        self.name+"/isqrt_lut", torch.tensor(inverse_sqrt_table), dtype=Dtype.INT16)
    self.constants["lut"].dtype = Dtype.INT16


@op_register(OpType.RMSNorm)
def RMSNorm(self, *args):
    axis = self.get_param('axis')
    eps = self.get_param('epsilon')

    inp = self.inputs[0]
    inp_betensor = inp.betensor.long() if self.quantized else inp.betensor.float()
    out = self.outputs[0]
    dim_num = inp_betensor.dim()
    channel_axis = dim_num - 1

    in_chunks = [inp_betensor]
    default_axis_reshape = [inp_betensor.shape[ax] if ax in axis else 1 for ax in range(dim_num)]
    axis_reshape = self.get_param('axis_shape', optional=True, default_value=default_axis_reshape)
    default_axis_reshape = [inp_betensor.shape[ax] if ax == axis[0] else 1 for ax in range(dim_num)]
    scale_shift_shape = self.get_param('scale_shift_shape', optional=True, default_value=default_axis_reshape)

    ngamma_chunks = []
    for t in in_chunks:
        if self.quantized:
            t_var = torch.mean((t.double()+inp.zerop)**2, dim=axis, keepdim=True).long()
            ngamma_qmin, ngamma_qmax = bits2range(16, is_signed=False)
            t_ngamma = calculate_inverse_sqrt(t_var)
            t_ngamma = (t_ngamma+32768) >> 16
            t_ngamma = linear_requantize(
                t_ngamma, self.params['norm_scale_value'], self.params['norm_shift_value'], 0, ngamma_qmin, ngamma_qmax)
            ngamma_chunks.append(t_ngamma)
        else:
            t_var = torch.mean((t.float())**2, dim=axis, keepdim=True)
            t_std = torch.pow(t_var + eps, 0.5)
            if torch.count_nonzero(t_std) != t_std.numel():
                OPT_WARN('type=%s, input std contain zero value, please check the axis or set epsilon nonzero value' % (str(self.type)))
            t_ngamma = 1.0 / (t_std)

            ngamma_chunks.append(t_ngamma)

        ngamma = torch.cat(ngamma_chunks, dim=channel_axis)

    if self.quantized:
        qmin, qmax = bits2range(out.qbits, is_signed=True)
        normalized = (inp_betensor + inp.broadcast_zerop) * ngamma
        normalized = torch.clip((normalized+128) >> 8, qmin, qmax)

    else:
        normalized = (inp_betensor) * ngamma
    out.betensor = normalized
    gamma = 1.0
    if 'weights' in self.constants:
        w = self.constants["weights"]
        # must be positive axis
        w.key_axis = len(w.shape) - 1
        gamma = w.betensor + w.broadcast_zerop
        gamma = torch.reshape(gamma, axis_reshape)
    beta = 0.0
    if 'biases' in self.constants:
        b = self.constants["biases"]
        b.key_axis = len(b.shape)-1
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

    return out.betensor
