# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.utils.quant_tool_utils import *
from AIPUBuilder.Optimizer.utils.string_utils import *
from AIPUBuilder.Optimizer.utils.math_utils import *
from AIPUBuilder.Optimizer.framework import *

import torch.nn as nn


def convert_less_mbit(data, threshold_min, threshold_max):
    left_shift = 0
    while (data > threshold_max) or (data < threshold_min):
        data = data >> 1
        left_shift += 1
    return data, left_shift


@op_register(OpType.Reduce)
def reduce(self, *args):
    inp = self.inputs[0]
    method = self.get_param('method').upper()
    axis = self.get_param('axis')
    outp = torch.ones_like(
        self.outputs[0].betensor, device=inp.betensor.device)
    if method == 'MIN':
        outp = torch.amin(inp.betensor, axis, keepdims=True)
    elif method == 'MAX':
        outp = torch.amax(inp.betensor, axis, keepdims=True)
    elif method == 'ANY':
        if self.quantized:
            inp.betensor += int(inp.zerop)
        outp = inp.betensor.bool()
        for i in axis:
            outp = torch.any(outp, i, keepdims=True)
        # output only is 0 or 1
        outp = outp.type(dtype2torch_type(self.outputs[0].dtype))
    elif method == 'ALL':
        if self.quantized:
            inp.betensor += int(inp.zerop)
        outp = inp.betensor.bool()
        for i in axis:
            outp = torch.all(outp, i, keepdims=True)
        # output only is 0 or 1
        outp = outp.type(dtype2torch_type(self.outputs[0].dtype))
    elif method == 'SUM':
        if self.quantized:
            outp = torch.sum(inp.betensor + int(inp.zerop), axis, keepdims=True)
            shift = self.params["shift_value"]
            scale = self.params["scale_value"]
            outp = linear_requantize(
                outp, scale, shift, self.outputs[0].zerop, self.outputs[0].qmin, self.outputs[0].qmax)
        else:
            outp = torch.sum(inp.betensor, axis, keepdims=True)
    elif method == 'MEAN':
        '''
        qx=scale*fp_x
        mean_qx = (scale*fp_x1+scale*fp_x2...+scale*fp_xn)/n = (fp_x1+fp_x2+fp_xn)/n*scale

        '''
        inp_betensor = inp.betensor + (torch_tensor(0, device=inp.device)
                                       if not self.quantized else self.inputs[0].zerop)
        outp = torch.sum(inp_betensor, axis, keepdims=True)
        if self.quantized:
            shift = self.params["shift_value"]
            scale = self.params["scale_value"]
            outp = linear_requantize(
                outp, scale, shift, self.outputs[0].zerop, self.outputs[0].qmin, self.outputs[0].qmax)
        else:
            inshape = inp_betensor.numel()
            outshape = outp.numel()
            div_dim = inshape / outshape
            outp = outp / div_dim
            # outp = torch.mean(inp.betensor, axis, keepdims=True)
    elif method == 'PROD':
        if self.quantized:
            inp_betensor = (inp.betensor + int(inp.zerop)).long()
            shift = self.params["shift_value"]
            scale = self.params["scale_value"]
            threshold_min, threshold_max = bits2range(16, is_signed(inp.dtype))
            inp_bak = inp_betensor.clone()
            input_dim = inp_betensor.dim()  # 4
            input_shape = list(inp_betensor.shape)
            outp = None
            step = 1
            current_transpose_shape = []
            for ax in range(input_dim):
                if ax in axis:
                    step *= input_shape[ax]
                    input_shape[ax] = 1
                else:
                    current_transpose_shape.append(ax)
            # shape[dim1,axis1,axis2,dim4] => shape[dim1,dim4,axis1,axis2]
            current_transpose_shape += axis
            inp_bak_transpose = inp_bak.permute(
                current_transpose_shape).reshape(-1,)
            outer_step = inp_bak.numel() // step
            outp_iter = torch.zeros((outer_step), device=inp_betensor.device)
            for i in range(outer_step):
                start_idx = i * step
                end_idx = i * step + step
                tmp_data = 1
                left_shifts = 0
                right_shifts = 0
                # (INT16) * (INT16) => INT32 => QINT6 * (2**left_shift)
                # (UINT16) * (UINT16) => UINT32 => QUINT6 * (2**left_shift)
                for j in range(start_idx, end_idx):
                    tmp_data *= inp_bak_transpose[j]
                    tmp_data, left_shift = convert_less_mbit(
                        tmp_data, threshold_min, threshold_max)
                    left_shifts += left_shift
                shift_diff = shift - left_shifts
                tmp_data = linear_requantize(
                    tmp_data, scale, shift_diff, self.outputs[0].zerop, self.outputs[0].qmin, self.outputs[0].qmax)
                outp_iter[i] = tmp_data
            outp = outp_iter.reshape(input_shape)
        else:
            outp = inp.betensor
            for i in axis:
                outp = torch.prod(outp, i, keepdim=True)
    elif method in ['VARIANCE', 'UNBIASED_VARIANCE']:
        input_data = inp.betensor.float()
        t_mean = torch.mean(input_data, dim=axis, keepdim=True)
        if self.quantized:
            t_mean = t_mean.round()
        tmp_square = (input_data - t_mean) ** 2
        tmp_sum = torch.sum(tmp_square, dim=axis, keepdim=True)

        if self.quantized:
            shift = self.params["shift_value"]
            scale = self.params["scale_value"]
            outp = linear_requantize(
                tmp_sum, scale, shift, self.outputs[0].zerop, self.outputs[0].qmin, self.outputs[0].qmax)
        else:
            num = 1
            for t in axis:
                num *= inp.betensor.shape[t]
            if method == "UNBIASED_VARIANCE":
                num = num - 1
            outp = tmp_sum / num
    elif method == 'L1':
        inp_betensor = inp.betensor
        if self.quantized:
            inp_betensor += int(self.inputs[0].zerop)
            inp_betensor = inp_betensor.long()

        abs_tensor = torch.abs(inp_betensor)
        outp = torch.sum(abs_tensor, axis, keepdim=True)
        if self.quantized:
            shift = self.params["shift_value"]
            scale = self.params["scale_value"]

            pmin, pmax = dtype2range(Dtype.UINT16)
            act_min, act_max = 0, 2**32-1

            # to be consistent with lib, take 16 higher bits
            # currently use uint32 to storage sum_output
            outp = torch.clamp(outp, act_min, act_max)
            sum_output_max = outp.max().float()
            max_bits = torch.ceil(torch.log2(sum_output_max)).item()
            shift_diff = 0
            if max_bits > 16:
                shift_diff = int(max_bits - 16)
            outp = outp >> shift_diff
            outp = torch.clamp(outp, pmin, pmax)
            outp = linear_requantize(
                outp, scale, shift-shift_diff, self.outputs[0].zerop, self.outputs[0].qmin, self.outputs[0].qmax)

    elif method == 'L2':
        inp_betensor = inp.betensor
        if self.quantized:
            inp_betensor += int(self.inputs[0].zerop)
            inp_betensor = inp_betensor.long()

        power2_input = inp_betensor * inp_betensor
        sum_output = torch.sum(power2_input, axis, keepdim=True)

        if self.quantized:
            shift = self.params["shift_value"]
            scale = self.params["scale_value"]
            sqrt_lut = self.constants['sqrt_lut'].betensor

            pmin, pmax = dtype2range(Dtype.UINT16)
            act_min, act_max = 0, 2**32-1

            # to be consistent with lib, take 16 higher bits
            # currently use uint32 to storage sum_output
            sum_output = torch.clamp(sum_output, act_min, act_max)
            sum_output_max = sum_output.max().float()
            max_bits = torch.ceil(torch.log2(sum_output_max)).int().item()
            shift_diff = 0
            if max_bits > 16:
                shift_diff = max_bits - 16
            sum_output = sum_output.long() >> shift_diff
            sum_output = torch.clamp(sum_output, pmin, pmax)
            sum_output = linear_requantize(
                sum_output, scale, shift-shift_diff, 0, pmin, pmax)
            sum_output_f = torch.reshape(sum_output, (-1,))
            sqrt = lookup_lut_powerof2(sum_output_f, sqrt_lut, 16, False, dtype2bits(
                self.constants["sqrt_lut"].dtype), is_signed(self.constants["sqrt_lut"].dtype))
            outp = torch.reshape(sqrt, sum_output.shape)
        else:
            if len(self.placeholders) < 1:
                ph0 = PyTensor(self.name+"/power2_sum",
                               sum_output.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
                self.placeholders.append(ph0)
            self.placeholders[0].betensor = sum_output
            outp = torch.sqrt(sum_output)
    else:
        OPT_FATAL("Reduce method is invalid")

    self.outputs[0].betensor = outp
    return outp


def getOutputSigned(inp_signed, method):
    if method in ['ANY', 'ALL', 'VARIANCE', 'UNBIASED_VARIANCE', 'L1', 'L2']:
        return False
    elif method in ['MIN', 'MAX', 'SUM', 'PROD', 'MEAN']:
        return inp_signed
    else:
        return True


@quant_register(OpType.Reduce)
def reduce_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    axis = self.get_param('axis')
    shift_type = SHIFT_DTYPE

    q_bits_activation = self.attrs["q_bits_activation"]
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    method = self.get_param('method').upper()
    out_signed = getOutputSigned(is_signed(inp.dtype), method) or self.force_dtype_int
    if method in ['ANY', 'ALL']:
        out.scale = 1
        out.zerop = 0
        out.qbits = 8
        out.dtype = bits2dtype(out.qbits, out_signed)
        out.qinvariant = True
    elif method in ['MIN', 'MAX']:
        out.scale = inp.scale
        out.zerop = inp.zerop
        out.qbits = inp.qbits
        out.dtype = bits2dtype(out.qbits, out_signed)
        out.qinvariant = inp.qinvariant
    else:
        if inp.qinvariant:
            out.scale = 1
            out.zerop = 0
            out.qbits, out.dtype = range2dtype(
                out.extrema_min, out.extrema_max, force_int=out_signed)
            out.qinvariant = True
            out.qmin, out.qmax = dtype2range(out.dtype)
            do_scale, do_shift = 1, 0
            do_scale_type = bits2dtype(out.qbits, False)
        else:
            # SUM, PROD operation need to quantize
            if method in ['SUM', 'PROD']:
                out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
                    out, q_mode_activation, q_bits_activation, is_signed=out_signed)
                out.qbits = q_bits_activation
                out.qinvariant = inp.qinvariant
                if method == 'SUM':
                    local_scale = out.scale / inp.scale
                    do_scale, do_scale_type, do_shift, do_shift_type = \
                        get_scale_approximation_params(local_scale,
                                                       mult_bits=q_bits_activation,
                                                       force_shift_positive=self.force_shift_positive)
                elif method == 'PROD':
                    input_scale = 1 / inp.scale  # (scale_x) / (shift_x)
                    local_scale = out.scale
                    q_bits_activation = max(16, q_bits_activation)
                    input_do_scale, input_do_scale_type, input_do_shift, input_do_shift_type = \
                        get_scale_approximation_params(input_scale,
                                                       mult_bits=q_bits_activation,
                                                       force_shift_positive=self.force_shift_positive)
                    do_scale, do_scale_type, do_shift, do_shift_type = \
                        get_scale_approximation_params(local_scale,
                                                       mult_bits=q_bits_activation,
                                                       force_shift_positive=self.force_shift_positive)
                    input_do_scale, input_do_shift = int(
                        input_do_scale), int(input_do_shift)
                    do_scale, do_shift = int(do_scale), int(do_shift)
                    input_shape = list(inp.ir_shape)
                    threshold_min, threshold_max = bits2range(15, False)
                    # input_do_scale ** n * do_scale = Q16 * 2 ** left_shifts
                    step = 1
                    for ax in axis:
                        step *= input_shape[ax]
                    tmp_data = 1
                    left_shifts = 0
                    for s in range(step):
                        tmp_data *= input_do_scale
                        tmp_data, left_shift = convert_less_mbit(
                            tmp_data, threshold_min, threshold_max)
                        left_shifts += left_shift
                    tmp_data, left_shift = convert_less_mbit(
                        tmp_data * do_scale, threshold_min, threshold_max)
                    left_shifts += left_shift
                    do_scale = tmp_data
                    do_shift = input_do_shift * step + do_shift - left_shifts
                    shift_type = Dtype.UINT32
            elif method in ['VARIANCE', 'UNBIASED_VARIANCE']:
                inp = self.inputs[0]
                out = self.outputs[0]
                out.qinvariant = False
                out.qbits = q_bits_activation
                out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
                    out, q_mode_activation, out.qbits, out_signed)
                num = 1
                for t in axis:
                    num *= inp.ir_shape[t]
                if method == "UNBIASED_VARIANCE":
                    num = num - 1
                do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(out.scale / (inp.scale * inp.scale) / num,
                                                                                                  mult_bits=16,
                                                                                                  shift_bits_ceil=63,
                                                                                                  force_shift_positive=self.force_shift_positive)
            elif method == "L1":
                out.qbits = q_bits_activation
                out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
                    out, q_mode_activation, q_bits_activation, is_signed=out_signed)
                out.qinvariant = False

                do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
                    out.scale/inp.scale, mult_bits=16, shift_bits_ceil=63)
                shift_type = do_shift_type

            elif method == "L2":
                placeholders = self.placeholders[0]
                placeholders.scale, placeholders.zerop, placeholders.qmin, placeholders.qmax, placeholders.dtype = \
                    get_linear_quant_params_from_tensor(
                        placeholders, QuantMode.to_symmetric(q_mode_activation), 16, is_signed=False)

                out.qbits = q_bits_activation
                out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
                    out, q_mode_activation, q_bits_activation, is_signed=out_signed)
                out.qinvariant = False

                do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
                    placeholders.scale/(inp.scale * inp.scale), mult_bits=16, shift_bits_ceil=63)
                shift_type = do_shift_type

                lsteps = 2 ** min(inp.qbits,
                                  int(self.get_attrs('lut_items_in_bits')))
                lut = linear_dequantize(torch.linspace(
                    placeholders.qmin, placeholders.qmax, steps=lsteps), placeholders.scale, 0)
                lut = torch.sqrt(lut)
                lut = linear_quantize_clip(
                    lut, out.scale, out.zerop, out.qmin, out.qmax)

                self.constants["sqrt_lut"] = PyTensor(
                    self.name+"/sqrt_lut", lut.cpu().numpy().astype(dtype2nptype(out.dtype)))

            else:  # method mean
                inshape = 1
                for s in inp.ir_shape:
                    inshape *= s
                outshape = 1
                for s in out.ir_shape:
                    outshape *= s
                div_dim = inshape / outshape
                local_scale = 1 / div_dim

                out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
                    out, q_mode_activation, q_bits_activation, is_signed=out_signed)
                out.qbits = q_bits_activation
                out.qinvariant = False
                local_scale *= out.scale / inp.scale
                do_scale, do_scale_type, do_shift, do_shift_type = \
                    get_scale_approximation_params(local_scale,
                                                   mult_bits=q_bits_activation,
                                                   force_shift_positive=self.force_shift_positive)
        self.params["shift_value"] = int(do_shift)
        self.params["shift_type"] = shift_type
        self.params["scale_value"] = int(do_scale)
        self.params["scale_type"] = do_scale_type
