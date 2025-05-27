# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.utils.quant_tool_utils import *
from AIPUBuilder.Optimizer.utils.string_utils import *
from AIPUBuilder.Optimizer.utils.math_utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.ops.cast import forward_with_clip

import torch.nn as nn

register_optype('Cumulate')


def convert_less_mbit(value, threshold_min, threshold_max):
    data = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.int64)
    left_shift = torch.zeros_like(data, device=data.device)
    while (data.max() > threshold_max) or (data.min() < threshold_min):
        max_mask = data > threshold_max
        min_mask = data < threshold_min
        data[max_mask] = (data[max_mask].long() >> 1).to(data.dtype)
        data[min_mask] = (data[min_mask].long() >> 1).to(data.dtype)
        left_shift[max_mask] += 1
        left_shift[min_mask] += 1
    if data.dim() < 1:
        return data.item(), left_shift.item()
    return data, left_shift


@op_register(OpType.Cumulate)
def cumulate(self, *args):
    input_data = self.inputs[0].betensor
    input_data = input_data + (torch.tensor(0, device=input_data.device)
                               if not self.quantized else torch.tensor(self.inputs[0].zerop, device=input_data.device))
    method = self.get_param('method').upper()
    axis = [int(self.get_param('axis'))]
    exclusive = self.get_param('exclusive')
    reverse = self.get_param('reverse')
    dev = input_data.device

    input_shape = list(input_data.shape)
    dim = input_data.dim()
    dim_list = [ax for ax in range(dim)]
    axis = [ax if ax >= 0 else ax+dim for ax in axis]
    step = 1
    axis_num = len(axis)
    pre_transpose_dim = []
    for ax in dim_list:
        if ax in axis:
            step *= input_shape[ax]
        else:
            pre_transpose_dim.append(ax)
    pre_transpose_dim = pre_transpose_dim + axis
    pre_in_shape = [input_shape[pre_transpose_dim[ax]] for ax in range(dim-axis_num)] + [step]
    post_transpose_dim = [pre_transpose_dim.index(ax) for ax in range(dim)]
    outer_step = input_data.numel() // step
    outp = torch.zeros([outer_step, step], device=input_data.device)

    input_transpose = input_data.permute(pre_transpose_dim).reshape([outer_step, step])
    if reverse:
        input_transpose = torch.flip(input_transpose, dims=[1])
    if method == 'SUM':
        if exclusive:
            input_transpose = torch.cat([torch.zeros([outer_step, 1], device=dev), input_transpose[:, :step-1]], dim=1)
        tmp_data = torch.zeros([outer_step], device=input_data.device)
        if self.quantized:
            scale = self.params['scale_value']
            shift = self.params['shift_value']
        for s in range(step):
            data_step = input_transpose[:, s]
            tmp_data = tmp_data + data_step
            outp[:, s] = tmp_data
            if self.quantized:
                outp[:, s] = linear_requantize(outp[:, s], scale, shift, self.outputs[0].zerop,
                                               self.outputs[0].qmin, self.outputs[0].qmax)
    elif method == 'PROD':
        if exclusive:
            input_transpose = torch.cat([torch.ones([outer_step, 1], device=dev), input_transpose[:, :step-1]], dim=1)
        tmp_data = torch.ones([outer_step], device=input_data.device)
        left_shifts = torch.zeros([outer_step], device=input_data.device)
        if self.quantized:
            scale = self.constants["scale"].betensor
            shift = self.constants["shift"].betensor
            threshold_min, threshold_max = bits2range(16, is_signed(self.inputs[0].dtype))

            for s in range(step):
                data_step = input_transpose[:, s]
                tmp_data = tmp_data * data_step
                tmp_data, data_left_shift = convert_less_mbit(tmp_data.long(), threshold_min, threshold_max)
                left_shifts += data_left_shift
                outp[:, s] = linear_requantize(tmp_data, scale[s], (shift[s]-left_shifts),
                                               self.outputs[0].zerop, self.outputs[0].qmin, self.outputs[0].qmax)
        else:
            for s in range(step):
                data_step = input_transpose[:, s]
                tmp_data = tmp_data * data_step
                outp[:, s] = tmp_data
    else:
        OPT_FATAL("unsupported method: %s for Cumulate in node:%s" % (method, self.name))

    if reverse:
        outp = torch.flip(outp, dims=[1])
    outp = torch.reshape(outp, pre_in_shape).permute(post_transpose_dim)

    if not self.quantized:
        outp = forward_with_clip(outp, self.outputs[0].dtype, 'TRUNCATION')
    self.outputs[0].betensor = outp
    return outp


@quant_register(OpType.Cumulate)
def cumulate_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    axis = [int(self.get_param('axis'))]
    method = self.get_param('method').upper()
    exclusive = self.get_param('exclusive')
    reverse = self.get_param('reverse')

    q_bits_activation = self.attrs["q_bits_activation"]
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")

    out_signed = is_signed(inp.dtype) or self.force_dtype_int
    out.qbits = inp.qbits  # q_bits_activation
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, is_signed=out_signed)
    out.qinvariant = False

    if inp.qinvariant:
        out.scale = 1
        out.zerop = 0
        out.qbits, out.dtype = range2dtype(
            out.extrema_min, out.extrema_max, force_int=out_signed)
        out.qinvariant = True
        out.qmin, out.qmax = dtype2range(out.dtype)
    if method == 'SUM':
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(out.scale/inp.scale,
                                           mult_bits=15,
                                           force_shift_positive=self.force_shift_positive)
        self.params["shift_value"] = int(do_shift)
        self.params["shift_type"] = do_shift_type
        self.params["scale_value"] = int(do_scale)
        self.params["scale_type"] = do_scale_type
    elif method == 'PROD':
        input_scale = 1 / inp.scale.item()
        local_scale = out.scale.item()
        input_shape = list(inp.ir_shape)
        step = 1
        for ax in axis:
            step *= input_shape[ax]
        input_scale, input_scale_type, input_shift, input_shift_type = \
            get_scale_approximation_params(input_scale,
                                           mult_bits=15,
                                           force_shift_positive=self.force_shift_positive)
        output_scale, out_scale_type, output_shift, out_shift_type = \
            get_scale_approximation_params(local_scale,
                                           mult_bits=15,
                                           force_shift_positive=self.force_shift_positive)

        do_scale = torch.zeros([step], device=inp.betensor.device)
        do_shift = torch.zeros([step], device=inp.betensor.device)
        left_shifts = 0
        current_scale = 1
        current_shift = 0
        scale_min, scale_max = bits2range(15, False)
        if exclusive:
            do_scale[0] = output_scale
            do_shift[0] = output_shift
        else:
            current_scale = input_scale
            current_shift = input_shift
            tmp_scale = current_scale * output_scale
            tmp_scale, out_scale_left_shift = convert_less_mbit(tmp_scale, scale_min, scale_max)
            do_scale[0] = tmp_scale
            do_shift[0] = current_shift + output_shift - out_scale_left_shift
        for s in range(1, step):
            current_scale *= input_scale
            current_shift += input_shift
            current_scale, input_scale_left_shift = convert_less_mbit(current_scale, scale_min, scale_max)
            left_shifts += input_scale_left_shift
            tmp_scale = current_scale * output_scale
            tmp_scale, out_scale_left_shift = convert_less_mbit(tmp_scale, scale_min, scale_max)
            do_scale[s] = tmp_scale
            do_shift[s] = current_shift + output_shift - left_shifts - out_scale_left_shift

        do_scale_type = Dtype.UINT16
        self.constants["scale"] = PyTensor(
            self.name+"/scale", do_scale.cpu().numpy().astype(dtype2nptype(do_scale_type)))
        self.constants["scale"].dtype = do_scale_type
        self.constants["shift"] = PyTensor(
            self.name+"/shift", do_shift.cpu().numpy().astype(dtype2nptype(Dtype.INT32)))
        self.constants["shift"].dtype = Dtype.INT32

    else:
        OPT_FATAL("unsupported method: %s for Cumulate in node:%s" % (method, self.name))
