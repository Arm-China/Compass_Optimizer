# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.activation import apply_with_activation, with_activation_out_is_signed, apply_with_activation_quantize, with_activation_allow_merge_out_zerop_to_bias
from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.ops.convwinograd import *
import torch
import math
import importlib


def _conv2d_torch_impl(self, *args):
    inp = self.inputs[0].betensor.float()
    weights = self.constants['weights'].betensor.clone().float()
    bias = self.constants['biases'].betensor.clone().float()
    stride = (self.get_param("stride_y"), self.get_param("stride_x"))
    dilation = (self.get_param('dilation_y'), self.get_param('dilation_x'))
    padding = (self.get_param('pad_left'), self.get_param('pad_right'),
               self.get_param('pad_top'), self.get_param('pad_bottom'))
    group = self.get_param('group')
    pad_val = 0
    if self.quantized:
        # input's zerop has been absorbed to bias.
        # inp += self.inputs[0].zerop
        pad_val = -self.inputs[0].zerop
        w_zp = self.constants["weights"].zerop
        w_zshape = [1] * weights.dim()
        w_zshape[0] = -1
        weights += w_zp.reshape(w_zshape) if isinstance(w_zp, torch.Tensor) else w_zp
        bias += self.constants['biases'].zerop

    inp = nhwc2nchw(inp)
    weights = nhwc2nchw(weights)
    if WinogradChecker.run(self):
        x = WinogradAllocator(self, inp, weights, bias)
        if self.get_param('with_winograd', optional=True, default_value=False):
            self.attrs['optimization_info']['with_winograd'] = True
    else:
        inp = torch.nn.functional.pad(inp, padding, value=pad_val)
        x = torch.nn.functional.conv2d(inp,
                                       weights,
                                       bias,
                                       stride=stride,
                                       padding=0,
                                       dilation=dilation,
                                       groups=group)
    x = nchw2nhwc(x)
    return x


@op_register(OpType.Convolution)
def conv2d(self, *args):
    in_shape = self.inputs[0].shape
    ih, iw = in_shape[1], in_shape[2]
    ky = self.get_param('kernel_y')
    kx = self.get_param('kernel_x')

    if not torch.cuda.is_available() and (ih >= 2048 and iw >= 2048) and (ky == 1 and kx == 1):
        OPT_DEBUG(f"when ih>=2048, iw>=2048 and ky==1, kw==1 with cpu device forward, we will use tf.nn.conv2d to execute.")
        conv2d_tf = importlib.import_module('AIPUBuilder.Optimizer.ops.tf_ops.conv2d')
        x = conv2d_tf.conv2d_tf_impl(self, *args)
    else:
        x = _conv2d_torch_impl(self, *args)

    requant_scale = 1
    requant_shift = 0
    if self.quantized:
        if 'scale_value' in self.params:
            requant_scale = self.params['scale_value']
        elif "scale" in self.constants:
            requant_scale = self.constants["scale"].betensor

        if 'shift_value' in self.params:
            requant_shift = self.params['shift_value']
        elif "shift" in self.constants:
            requant_shift = self.constants["shift"].betensor

    x = apply_with_activation(self,
                              x,
                              self.inputs[0].scale * self.constants["weights"].scale,
                              0,
                              requant_scale,
                              requant_shift,
                              *args)

    self.outputs[0].betensor = x
    return self.outputs[0].betensor


def unify_shifts_for_aiff(scales, shifts):
    scales = scales.long()
    shifts = shifts.long()
    m_shift = max(0, shifts.max().item())
    new_scales = scales * (2 ** (m_shift - shifts))  # [s * (1 << (m_shift - ss)) for s, ss in zip(scales, shifts)]
    # mask = 0xffff
    while new_scales.max().item() > ((1 << 15) - 1):
        new_scales = new_scales >> 1  # [s >> 1 for s in new_scales]
        m_shift -= 1
    return new_scales, torch.zeros_like(new_scales) + max(0, m_shift)


@quant_register(OpType.Convolution)
def conv2d_quantize(self, *args):
    if WinogradChecker.run(self):
        self.constants["weights"] = self.constants['WinogradWeights']
        linear_op_quantize(self, *args)
        self.constants.pop('WinogradWeights')
    else:
        linear_op_quantize(self, *args)
    absorb_input_zp_to_bias(self, *args)
    clear_lower_bits_for_bias(self, *args)


def linear_op_quantize(self, *args):
    q_mode_bias = self.attrs["q_mode_bias"]
    q_mode_weight = self.attrs["q_mode_weight"]
    q_mode_activation = self.attrs["q_mode_activation"]
    if q_mode_weight != q_mode_bias:
        OPT_FATAL("Currently quantization mode of weight (q_mode_weight) and bias (q_mode_bias) must be the same!")
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_weight = self.attrs["q_bits_weight"]
    q_bits_bias = self.attrs["q_bits_bias"]
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]

    w = self.constants["weights"]
    key_axis_c = w.key_axis_c
    w_out_cnum = w.shape[key_axis_c]
    w_scale_expand = []
    w_zerop_expand = []
    group = self.get_param('group', optional=True, default_value=1)

    out = self.outputs[0]
    out_signed = with_activation_out_is_signed(self) or self.force_dtype_int
    out.qbits = q_bits_activation

    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, q_bits_activation, is_signed=out_signed)
    out.qinvariant = False

    if group > 1 or QuantMode.is_per_channel(q_mode_weight):
        if QuantMode.is_per_channel(q_mode_weight):
            mgroup = w_out_cnum
        else:
            mgroup = group
        count = 0
        t = PyTensor(self.name + '/temp_var' + str(count))
        while mgroup >= 1:
            current_per_group_cnum = w_out_cnum // mgroup
            min_groups = torch.split(w.min_key_axis, current_per_group_cnum, dim=0)
            max_groups = torch.split(w.max_key_axis, current_per_group_cnum, dim=0)
            for i, min_values in enumerate(min_groups):
                max_values = max_groups[i]
                min_v = min_values.min().item()
                max_v = max_values.max().item()
                t.min = min_v
                t.max = max_v
                tmp_q_mode_weight = q_mode_weight.replace('channel', 'tensor')
                scale, zerop, w.qmin, w.qmax, w.dtype = get_linear_quant_params_from_tensor(
                    t, tmp_q_mode_weight, q_bits_weight, is_signed=True)
                w_scale_expand.extend([scale] * min_values.shape[0])
                w_zerop_expand.extend([zerop] * min_values.shape[0])

            w.scale = torch.tensor(w_scale_expand, device=inp.betensor.device)
            w.zerop = torch.tensor(w_zerop_expand, device=inp.betensor.device)
            local_rescale = out.scale / (inp.scale * w.scale)
            do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
                local_rescale, mult_bits=q_bits_activation, force_shift_positive=self.force_shift_positive)
            if self.attrs['unify_shifts_for_aiff']:
                new_scale, new_shift = unify_shifts_for_aiff(do_scale, do_shift)
                do_scale, do_shift = new_scale, new_shift
                # new_scale may narrow down to 0 after unify_shifts_for_aiff as there is no suitable results
                if new_scale.min().item() > 0 or mgroup == 1:
                    break
            else:
                break
            mgroup = mgroup >> 1
            count += 1
            w_scale_expand.clear()
            w_zerop_expand.clear()
        if count != 0:
            OPT_WARN('layer_id=%s, type=%s, due to hardware limitations, it is actually doing per-%s-channel, which may cause accuracy dropping'
                     % (self.attrs['layer_id'], str(self.type), str(w_out_cnum//mgroup)))

        _, do_scale_type = range2dtype(0, do_scale.max().item())
        _, do_shift_type = range2dtype(do_shift.min().item(), do_shift.max().item(), force_int=True)
        self.constants["scale"] = PyTensor(
            self.name+"/scale", do_scale.cpu().numpy().astype(dtype2nptype(do_scale_type)))
        self.constants["scale"].dtype = do_scale_type
        self.constants["shift"] = PyTensor(
            self.name+"/shift", do_shift.cpu().numpy().astype(dtype2nptype(do_shift_type)))
        self.constants["shift"].dtype = do_shift_type

    else:
        w.scale, w.zerop, w.qmin, w.qmax, w.dtype = get_linear_quant_params_from_tensor(
            w, q_mode_weight, q_bits_weight, is_signed=True)
        local_rescale = out.scale / (inp.scale * w.scale)
        do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
            local_rescale, mult_bits=q_bits_activation, force_shift_positive=self.force_shift_positive)
        w_scale_expand.extend([w.scale] * w_out_cnum)
        w_zerop_expand.extend([w.zerop] * w_out_cnum)
        _, do_scale_type = range2dtype(0, do_scale)
        self.params["shift_value"] = int(do_shift)
        self.params["shift_type"] = do_shift_type
        self.params["scale_value"] = int(do_scale)
        self.params["scale_type"] = do_scale_type

    # quantize weights
    w_scale_expand = torch.tensor(w_scale_expand, device=inp.betensor.device)
    w_zerop_expand = torch.tensor(w_zerop_expand, device=inp.betensor.device)
    scale_zp_shape = [w.shape[ax] if ax == key_axis_c else 1 for ax in range(len(w.shape))]
    w.betensor = linear_quantize_clip(w.betensor, w_scale_expand.reshape(scale_zp_shape),
                                      w_zerop_expand.reshape(scale_zp_shape), w.qmin, w.qmax)
    w.qbits = q_bits_weight
    w.qinvariant = False

    # quantize bias
    b = self.constants["biases"]
    # if self.get_param('with_activation', optional=True, default_value='none').lower() in with_activation_allow_merge_out_zerop_to_bias :
    #     #y_q = ((w_q + zp_w) (x_q + zp_x) + (b_q + zp_b)s_ws_x/s_b) s_y/s_ws_x - zp_y
    #     #y_q = ((w_q + zp_w) (x_q + zp_x) + (b_f - zp_y/s_y)s_ws_x) s_y/s_ws_x
    #     b.betensor = b.betensor - linear_dequantize(0, out.scale, out.zerop)
    b.scale = inp.scale * w.scale
    b.zerop = 0
    b.qmin = -2**(q_bits_bias-1)
    b.qmax = 2**(q_bits_bias-1) - 1
    b.qbits = q_bits_bias
    if isinstance(b.scale, torch.Tensor):
        b_scale_shape = [b.shape[ax] if ax == b.key_axis_c else 1 for ax in range(len(b.shape))]
        b.scale = torch.reshape(b.scale, b_scale_shape)
    b.betensor = linear_quantize_clip(b.betensor, b.scale, b.zerop, b.qmin, b.qmax)
    # AIFF will scale down biases to 16bit and rescale it back to 32bit later due to hardware restricts
    bmin = min(b.betensor.min().item(), -1)
    bmax = max(b.betensor.max().item(), 1)
    lmin, lmax = bits2range(self.attrs['bias_effective_bits'], is_signed=True)
    lbits = math.ceil(max(math.log2(bmax/lmax), math.log2(bmin/lmin)))
    if lbits > 0:
        b.betensor = ((b.betensor.long() >> lbits) << lbits).float()

    b.dtype = bits2dtype(b.qbits, is_signed=True)
    b.qinvariant = False

    apply_with_activation_quantize(self, out.qinvariant, *args)


def clear_lower_bits_for_bias(self, *args):
    # AIFF will scale down biases to 16bit and rescale it back to 32bit later due to hardware restricts
    b = self.constants["biases"]
    bmin = min(b.betensor.min().item(), -1)
    bmax = max(b.betensor.max().item(), 1)
    lmin, lmax = bits2range(self.attrs['bias_effective_bits'], is_signed=True)
    lbits = math.ceil(max(math.log2(bmax/lmax), math.log2(bmin/lmin)))
    if lbits > 0:
        b.betensor = ((b.betensor.long() >> lbits) << lbits).float()


def compute_input_zp_mul_reduce_weight(zp, w):
    return w.reshape(w.shape[0], -1).sum(dim=1) * zp


def absorb_input_zp_to_bias(self, *args):
    w = self.constants["weights"]
    b = self.constants["biases"]

    input_zp_mul_weight = compute_input_zp_mul_reduce_weight(self.inputs[0].zerop, w.betensor)
    input_zp_mul_weight = input_zp_mul_weight.item() if b.ir_shape == TensorShape([]) else input_zp_mul_weight
    b.betensor += input_zp_mul_weight
    b.betensor = b.betensor.clamp(b.qmin, b.qmax)
