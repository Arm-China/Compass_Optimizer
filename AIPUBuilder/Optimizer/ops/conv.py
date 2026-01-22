# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.ops.convwinograd import *
from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
from AIPUBuilder.Optimizer.ops.activation import (apply_with_activation,
                                                  with_activation_out_is_signed,
                                                  apply_with_activation_quantize,
                                                  with_activation_allow_merge_out_zerop_to_bias)
import torch
import math
import importlib


def _conv2d_torch_impl(self, *args):
    inp = self.inputs[0].betensor.double()
    weights = self.constants['weights'].betensor.clone().double()
    bias = self.constants['biases'].betensor.clone().double()
    stride = (self.get_param("stride_y"), self.get_param("stride_x"))
    dilation = (self.get_param('dilation_y'), self.get_param('dilation_x'))
    padding = (self.get_param('pad_left'), self.get_param('pad_right'),
               self.get_param('pad_top'), self.get_param('pad_bottom'))
    group = self.get_param('group')
    pad_val = 0
    if self.quantized:
        # input's zerop has been absorbed to bias.
        # inp += self.inputs[0].zerop
        pad_val = -self.inputs[0].zerop[0]
        weights += self.constants["weights"].broadcast_zerop
        bias += self.constants['biases'].broadcast_zerop

    inp = nhwc2nchw(inp)
    weights = nhwc2nchw(weights)
    if WinogradChecker.run(self) or self.get_param('with_winograd', optional=True, default_value=False):
        x = WinogradAllocator(self, inp, weights, bias)
    else:
        if self.aasrb is not None:
            bias = None
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

    self.aasrb = self.get_param('remain_shift', optional=True, default_value=None)

    if not torch.cuda.is_available() and (ih >= 2048 and iw >= 2048) and (ky == 1 and kx == 1):
        OPT_DEBUG(f"when ih>=2048, iw>=2048 and ky==1, kw==1 with cpu device forward, we will use tf.nn.conv2d to execute.")
        conv2d_tf = importlib.import_module('AIPUBuilder.Optimizer.ops.tf_ops.conv2d')
        x = conv2d_tf.conv2d_tf_impl(self, *args)
    else:
        x = _conv2d_torch_impl(self, *args)

    shift_bk = None
    if self.quantized and self.aasrb is not None and (dtype2bits(self.constants["weights"].dtype) > 8 or dtype2bits(self.inputs[0].dtype) > 8):
        bias = self.constants['biases'].betensor.clone().float() + self.constants['biases'].broadcast_zerop
        self.outputs[0].betensor = apply_with_activation(self, x,
                                                         *args, aasrb=(self.aasrb, bias))
        return self.outputs[0].betensor
    self.outputs[0].betensor = apply_with_activation(self, x, *args)
    return self.outputs[0].betensor


def fpx_weight_quantize(self, *args):
    quant_type = self.attrs.get('quant_type')
    q_type_weight = QuantType._to_Dtype(QuantType.weight_type(quant_type))
    q_mode_weight = self.attrs["q_mode_weight"]

    w = self.constants["weights"]
    key_axis = w.key_axis
    w_out_cnum = w.shape[key_axis]
    group = self.get_param('group', optional=True, default_value=1)

    w.dtype = q_type_weight
    w.qbits = dtype2bits(w.dtype)
    w.qinvariant = False

    # if activation_type is fp8/fp4(ab. fpx), weight_type support int4 or fpx
    if (q_type_weight == Dtype.ALIGNED_INT4) or (q_type_weight == Dtype.FP4_E2M1FN):
        # GPTQ quantization, scale0 and scale1 are necessary
        if all(sname in self.constants.keys() for sname in ['scale0', 'scale1']):
            w.qmin, w.qmax = dtype2range(w.dtype)
        else:
            # opt do quantization
            w.scale, w.zerop, w.qmin, w.qmax, _ = get_linear_quant_params_from_tensor(
                w, q_mode_weight, w.qbits, is_signed=True)
            w.betensor = linear_quantize_clip(w.betensor, w.broadcast_scale, w.broadcast_zerop, w.qmin, w.qmax,
                                              round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', w.dtype))
    else:
        if group > 1 or QuantMode.is_per_channel(q_mode_weight):
            if QuantMode.is_per_channel(q_mode_weight):
                initial_group = w_out_cnum
            else:
                initial_group = group
            snum = w.min_key_axis.numel()
            mgroup = min(initial_group, snum)
            current_per_group_cnum = math.ceil(float(snum) / float(mgroup))
            t = PyTensor('temp_var_unify_shifts_for_aiff_with_per_n_channel_quant')

            current_per_group_cnum = math.ceil(float(snum) / float(mgroup))
            current_pad_cnum = mgroup * current_per_group_cnum - snum
            t.min_key_axis = torch.nn.functional.pad(w.min_key_axis, (0, current_pad_cnum), mode="constant",
                                                     value=0.).reshape([-1, current_per_group_cnum]).min(dim=1).values
            t.max_key_axis = torch.nn.functional.pad(w.max_key_axis, (0, current_pad_cnum), mode="constant",
                                                     value=0.).reshape([-1, current_per_group_cnum]).max(dim=1).values

            w.scale, w.zerop, w.qmin, w.qmax = get_fpx_quant_params_from_tensor(t, q_mode_weight, w.dtype)

            w.scale = w.scale.repeat_interleave(current_per_group_cnum)[:snum]
            w.zerop = w.zerop.repeat_interleave(current_per_group_cnum)[:snum]
        else:
            w.scale, w.zerop, w.qmin, w.qmax = get_fpx_quant_params_from_tensor(w, q_mode_weight, q_type_weight)
        w.betensor = linear_quantize_clip(w.betensor.float(), w.broadcast_scale, w.broadcast_zerop, w.qmin, w.qmax,
                                          round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', w.dtype))


def conv2d_fpx_quantize(self, *args):
    quant_type = self.attrs.get('quant_type')
    q_type_activation = QuantType._to_Dtype(QuantType.activation_type(quant_type))
    q_type_weight = QuantType._to_Dtype(QuantType.weight_type(quant_type))
    q_type_bias = QuantType._to_Dtype(QuantType.bias_type(quant_type))

    q_mode_weight = self.attrs["q_mode_weight"]
    q_mode_activation = self.attrs["q_mode_activation"]
    q_bits_weight = self.attrs["q_bits_weight"]
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    out = self.outputs[0]
    # key_axis = w.key_axis
    # w_out_cnum = w.shape[key_axis]
    # group = self.get_param('group', optional=True, default_value=1)

    out.qbits = q_bits_activation
    out.dtype = q_type_activation
    out.scale, out.zerop, out.qmin, out.qmax = get_fpx_quant_params_from_tensor(
        out, q_mode_activation, q_type_activation)
    out.qinvariant = False

    fpx_weight_quantize(self, *args)
    w = self.constants["weights"]

    wscale = w.scale
    if all(sname in self.constants.keys() for sname in ['scale0', 'scale1']):
        wscale = 1.0
    local_rescale = out.scale / (inp.scale * wscale)
    do_scale = local_rescale
    do_scale_type = Dtype.FP32

    doscale_name = 'scale' if is_torch_tensor_with_multi_data(do_scale) else 'scale_value'
    self.set_ir_field(doscale_name, do_scale, do_scale_type)
    if not is_torch_tensor_with_multi_data(do_scale):
        self.params["scale_type"] = do_scale_type

    if 'biases' in self.constants:
        b = self.constants["biases"]
        # if self.get_param('with_activation', optional=True, default_value='none').lower() in with_activation_allow_merge_out_zerop_to_bias :
        #     #y_q = ((w_q + zp_w) (x_q + zp_x) + (b_q + zp_b)s_ws_x/s_b) s_y/s_ws_x - zp_y
        #     #y_q = ((w_q + zp_w) (x_q + zp_x) + (b_f - zp_y/s_y)s_ws_x) s_y/s_ws_x
        #     b.betensor = b.betensor - linear_dequantize(0, out.scale, out.zerop)
        b.scale = inp.scale * wscale
        b.zerop = 0
        b.dtype = q_type_bias
        b.qmin, b.qmax = dtype2range(b.dtype)
        b.qbits = dtype2bits(b.dtype)
        b.betensor = linear_quantize_clip(b.betensor.float(), b.broadcast_scale, b.broadcast_zerop,
                                          b.qmin, b.qmax, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', b.dtype))
        b.qinvariant = False
    bk_inp_scale = self.inputs[0].scale
    self.inputs[0].scale = inp.scale
    apply_with_activation_quantize(self, out.qinvariant, *args)
    self.inputs[0].scale = bk_inp_scale


@quant_register(OpType.Convolution)
def conv2d_quantize(self, *args):
    quant_type = self.attrs.get('quant_type')
    fpx_quantize = True if quant_type != 'disable' and QuantType.is_float(quant_type) else False
    quantize_func = conv2d_fpx_quantize if fpx_quantize else linear_op_quantize
    if WinogradChecker.run(self):
        self.constants["weights"] = self.constants['WinogradWeights']
        quantize_func(self, *args)
        self.constants.pop('WinogradWeights')
        if self.get_param('with_winograd', optional=True, default_value=False):
            self.attrs.update({'optimization_info': {'with_winograd': True}})
    else:
        quantize_func(self, *args)

    if not fpx_quantize:
        absorb_input_zp_to_bias_and_compress_bias_for_aiff(self, *args)
        if 'remain_shift' in self.attrs:
            self.params['remain_shift'] = self.attrs['remain_shift']


def linear_op_quantize(self, *args):
    q_mode_bias = self.attrs["q_mode_bias"]
    q_mode_weight = self.attrs["q_mode_weight"]
    q_mode_activation = self.attrs["q_mode_activation"]
    if q_mode_weight != q_mode_bias:
        OPT_FATAL("Currently quantization mode of weight (q_mode_weight) and bias (q_mode_bias) must be the same!")
    q_bits_weight = self.attrs["q_bits_weight"]
    q_bits_bias = self.attrs["q_bits_bias"]
    q_bits_activation = self.attrs["q_bits_activation"]
    multiplier_bits = self.get_attrs('multiplier_bits', optional=True, default_value=q_bits_activation)

    inp = self.inputs[0]

    w = self.constants["weights"]
    key_axis = w.key_axis
    w_out_cnum = w.shape[key_axis]
    group = self.get_param('group', optional=True, default_value=1)

    out = self.outputs[0]
    out_signed = with_activation_out_is_signed(self) or self.force_dtype_int
    out.qbits = q_bits_activation

    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, q_bits_activation, is_signed=out_signed)
    out.qinvariant = False

    inp_scale = inp.scale
    '''it is not convenient to import whitelist of act perchannel, so we put its to quant_tool_utils, as ABSORB_INPUT_SCALE_OP.'''
    if self.type in ABSORB_INPUT_SCALE_OP and inp.is_perchannel_scales():
        from AIPUBuilder.Optimizer.features import statistic_and_calibration
        OPT_DEBUG(f"{self} will absorbs input scale to weight in linear_op_quantize.", log_once=True)
        w.betensor /= inp_scale
        inp_scale = 1.0
        w.qbits = q_bits_weight
        statistic_and_calibration(w, self.attrs, is_constant_tensor=True)
    if not all(sname in self.constants.keys() for sname in ['scale0', 'scale1']):
        if group > 1 or QuantMode.is_per_channel(q_mode_weight):
            if QuantMode.is_per_channel(q_mode_weight):
                mgroup = w_out_cnum
            else:
                mgroup = group

            do_scale, do_scale_type, do_shift, do_shift_type = unify_shifts_for_aiff_with_per_n_channel_quant(
                self, w, q_mode_weight, q_bits_weight, True, lambda xt: out.scale / (inp_scale * xt.scale), initial_group=mgroup)

        else:
            w.scale, w.zerop, w.qmin, w.qmax, w.dtype = get_linear_quant_params_from_tensor(w, q_mode_weight, q_bits_weight,
                                                                                            is_signed=True)
            local_rescale = out.scale / (inp_scale * w.scale)
            do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(local_rescale,
                                                                                              mult_bits=multiplier_bits,
                                                                                              force_shift_positive=self.force_shift_positive)
        wscale, wzerop = w.scale, w.zerop
        # quantize weights
        w.betensor = linear_quantize_clip(w.betensor, w.broadcast_scale, w.broadcast_zerop, w.qmin, w.qmax)
        w.qbits = q_bits_weight
    else:
        wscale, wzerop = 1.0, 0.0
        w.dtype = Dtype.ALIGNED_INT4
        w.qbits = dtype2bits(w.dtype)
        local_rescale = out.scale / (inp_scale * wscale)
        do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(local_rescale,
                                                                                          mult_bits=multiplier_bits,
                                                                                          force_shift_positive=self.force_shift_positive)

    doscale_name = 'scale' if is_torch_tensor_with_multi_data(do_scale) else 'scale_value'
    doshift_name = 'shift' if is_torch_tensor_with_multi_data(do_shift) else 'shift_value'
    self.set_ir_field(doscale_name, do_scale, do_scale_type)
    self.set_ir_field(doshift_name, do_shift, do_shift_type)
    if not is_torch_tensor_with_multi_data(do_scale):
        self.params["shift_type"] = do_shift_type
        self.params["scale_type"] = do_scale_type
    w.qinvariant = False

    if 'biases' in self.constants:
        b = self.constants["biases"]
        # if self.get_param('with_activation', optional=True, default_value='none').lower() in with_activation_allow_merge_out_zerop_to_bias :
        #     #y_q = ((w_q + zp_w) (x_q + zp_x) + (b_q + zp_b)s_ws_x/s_b) s_y/s_ws_x - zp_y
        #     #y_q = ((w_q + zp_w) (x_q + zp_x) + (b_f - zp_y/s_y)s_ws_x) s_y/s_ws_x
        #     b.betensor = b.betensor - linear_dequantize(0, out.scale, out.zerop)
        b.scale = inp_scale * wscale
        b.zerop = 0

        if self.attrs['bias_effective_bits'] != q_bits_bias:
            search_bias_bit_and_quantize(self, *args)
        else:
            b.qmin = -2 ** (q_bits_bias - 1)
            b.qmax = 2 ** (q_bits_bias - 1) - 1
            b.qbits = q_bits_bias
            b.betensor = linear_quantize_clip(b.betensor, b.broadcast_scale, b.broadcast_zerop, b.qmin, b.qmax)

        b.dtype = bits2dtype(b.qbits, is_signed=True)
        b.qinvariant = False

    bk_inp_scale = self.inputs[0].scale
    self.inputs[0].scale = inp_scale
    apply_with_activation_quantize(self, out.qinvariant, *args)
    self.inputs[0].scale = bk_inp_scale


def clear_lower_bits_for_bias(self, *args, grouped=False, dim=-1):
    # AIFF will scale down biases to 16bit and rescale it back to 32bit later due to hardware restricts
    b = self.constants["biases"]
    lmin, lmax = bits2range(self.attrs['bias_effective_bits'], is_signed=True)
    if not grouped:
        bmin = min(b.betensor.min().item(), -1)
        bmax = max(b.betensor.max().item(), 1)
        lbits = math.ceil(max(math.log2(bmax/lmax), math.log2(bmin/lmin)))
        if lbits > 0:
            b.betensor = ((b.betensor.long() >> lbits) << lbits).float()
    else:
        bmin = torch.min(b.betensor, dim=dim)[0]
        bmax = torch.max(b.betensor, dim=dim)[0]
        lbits = torch.ceil(torch.maximum(torch.log2(bmax / lmax), torch.log2(bmin / lmin))).long().unsqueeze(dim)
        b.betensor = ((b.betensor.long() >> lbits) << lbits).float()


def compute_input_zp_mul_reduce_weight(zp, w):
    return (w * zp).reshape(w.shape[0], -1).sum(dim=1)


def absorb_input_zp_to_bias(self, *args):
    w = self.constants["weights"]
    b = self.constants["biases"]

    if self.get_attrs('is_winograd_checker_passed', optional=True, default_value=False):
        device = w.betensor.device
        BT, _, AT = ArcReactor.run(2, 3)
        BT = torch.from_numpy(BT).to(device)
        AT = torch.from_numpy(AT).to(device)
        input_zp = torch_tensor(self.inputs[0].zerop, device)
        weight = w.betensor.permute(0, 3, 1, 2)
        repeated_input_zp = input_zp.repeat([1, BT.shape[0]])  # pylint: disable=no-member
        data_tmp = torch.matmul(repeated_input_zp.float(), BT.permute(1, 0))
        data_repeat_shape = [*weight.shape[0:3]] + [1]
        BT_out = data_tmp.repeat(data_repeat_shape)
        BT_out = BT_out * weight
        AT_out = torch.matmul(BT_out, AT.permute(1, 0))
        input_zp_mul_weight = AT_out.sum(dim=[1, 2])[:, 0]
    else:
        input_zp_mul_weight = compute_input_zp_mul_reduce_weight(self.inputs[0].zerop, w.betensor)
        input_zp_mul_weight = input_zp_mul_weight.item() if b.ir_shape == TensorShape([]) else input_zp_mul_weight

    b.betensor += input_zp_mul_weight
    b.betensor = b.betensor.clamp(b.qmin, b.qmax)


def absorb_input_zp_to_bias_and_compress_bias_for_aiff(self, *args):
    absorb_input_zp_to_bias(self, *args)
    clear_lower_bits_for_bias(self, *args)


def search_bias_bit_and_quantize(self, *args):
    b = self.constants['biases']
    q_bits_bias = self.attrs["q_bits_bias"]
    bias_effective_bits = self.attrs['bias_effective_bits']
    if bias_effective_bits > q_bits_bias:
        OPT_WARN(f"layer_id = {self.attrs['layer_id']}, layer_type = {self.type}, layer_name = {self.name}",
                 f"the bias_effective_bits(={bias_effective_bits}) > bias_bits(={q_bits_bias}), "
                 f"we clip bias_effective_bits to bias_bits.")
        bias_effective_bits = q_bits_bias
        self.attrs['bias_effective_bits'] = bias_effective_bits

    bias_float_data = b.betensor.clone()
    b.qmin = -2 ** (q_bits_bias - 1)
    b.qmax = 2 ** (q_bits_bias - 1) - 1
    b.qbits = q_bits_bias
    b.betensor = linear_quantize_clip(b.betensor, b.scale, b.zerop, b.qmin, b.qmax)
    dev = b.device

    q_bias_data = b.betensor.clone()
    clear_lower_bits_for_bias(self)
    compressed_q_bias_data = b.betensor
    float_softmax = bias_float_data.clone().softmax(dim=-1)
    quantize_log_softmax = q_bias_data.log_softmax(dim=-1)
    compress_log_softmax = compressed_q_bias_data.log_softmax(dim=-1)
    float_quant_kld = torch.nn.functional.kl_div(quantize_log_softmax, float_softmax, reduction='batchmean')
    float_compress_kld = torch.nn.functional.kl_div(compress_log_softmax, float_softmax, reduction='batchmean')
    if (float_quant_kld - float_compress_kld).abs() > 0.05 and self.get_attrs('bias_effective_bits_auto_adaption', optional=True, default_value=False):
        begin_bits = min(max(bias_effective_bits, 19), q_bits_bias - 1)
        end_bits = q_bits_bias
        bits = torch.arange(begin_bits, end_bits, device=dev)
        repeated_bias = torch.tile(bias_float_data, tuple([end_bits - begin_bits] + [1 for _ in range(len(b.shape))]))
        clamp_min = -2 ** (bits - 1)
        clamp_max = 2 ** (bits - 1) - 1
        q_biases = linear_quantize_clip(repeated_bias,
                                        torch_tensor(
                                            b.scale, device=dev).unsqueeze(0),  # pylint: disable=no-member
                                        b.zerop,
                                        clamp_min.unsqueeze(-1),
                                        clamp_max.unsqueeze(-1))
        b.betensor = q_biases.clone()
        clear_lower_bits_for_bias(self, grouped=True)
        compress_biases = b.betensor
        q_softmax = linear_dequantize(q_biases, b.scale, b.zerop).softmax(dim=-1)
        cq_log_softmax = linear_dequantize(compress_biases, b.scale, b.zerop).log_softmax(dim=-1)
        klds = []
        for i in range(end_bits - begin_bits):
            kld = torch.nn.functional.kl_div(cq_log_softmax[i], q_softmax[i], reduction='batchmean').abs()
            klds.append(kld.item())

        searched_bias_bits = klds.index(min(klds)) + begin_bits
        b.qmin = -2 ** (searched_bias_bits - 1)
        b.qmax = 2 ** (searched_bias_bits - 1) - 1
        b.qbits = max(searched_bias_bits, q_bits_bias)
        b.betensor = linear_quantize_clip(bias_float_data, b.scale, b.zerop, b.qmin, b.qmax)
        OPT_DEBUG((f"layer_id = {self.attrs['layer_id']}, layer_type = {self.type}, layer_name = {self.name} "
                   f"searched bias bits = {searched_bias_bits} for better quantize bias "
                   f"to reduce the compression bias impact in AIFF"))
    else:
        b.betensor = q_bias_data
