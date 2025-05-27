# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import math
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import *


@quant_register(OpType.Softmax)
def softmax_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    q_bits_activation = self.attrs["q_bits_activation"]
    inp = self.inputs[0]
    dev = inp.betensor.device
    # so we allow to accumulate less than 2**12 items with a 32bit accumulator
    axis = self.get_param('axis')
    shape_value_in_axis = inp.ir_shape[axis]
    ibits = torch.ceil(torch.log2(torch.tensor(shape_value_in_axis))).long().item()
    extra_params = self.attrs['extra_params']

    out = self.outputs[0]

    out.qbits = q_bits_activation
    out.qinvariant = False

    smethod = int(extra_params[0] if len(extra_params) > 0 else 1)
    if QuantMode.is_per_channel(q_mode_activation) == True and 2 == smethod:
        OPT_FATAL("Softmax currently not support per-channel quantization with extra_params[0] == 2")

    # Check feature map size
    shape = inp.ir_shape
    h, w, c = 0, 0, 0
    if shape.dim() == 2:
        h = w = shape[0]
        c = shape[1]
    elif shape.dim() == 3:
        h, w = shape[:2]
        c = shape[2]
    elif shape.dim() == 4:
        h, w = shape[1:3]
        c = shape[-1]

    min_compatible_zhouyi_target = self.attrs["min_compatible_zhouyi_target"].upper()
    if 1 == smethod:
        adjust_q = int(extra_params[1] if len(extra_params) > 1 else 1)
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(1.442695 / inp.scale,
                                           mult_bits=13,
                                           force_shift_positive=self.force_shift_positive)
        scale_name = 'input_scale' if is_torch_tensor_with_multi_data(do_scale) else 'input_scale_value'
        shift_name = 'input_shift' if is_torch_tensor_with_multi_data(do_shift) else 'input_shift_value'
        self.set_ir_field(scale_name, do_scale, do_scale_type)
        self.set_ir_field(shift_name, do_shift, do_shift_type)
        if scale_name == 'input_scale_value':
            self.params["input_scale_type"] = bits2dtype(16, is_signed=False)
            self.params["input_shift_type"] = do_shift_type

        self.params["pow2_out_q"] = 31 - torch.floor(torch.log2(torch.tensor(shape_value_in_axis))).long()
        self.outputs[0].max = 1.0
        if QuantMode.is_per_channel(q_mode_activation):
            self.outputs[0].max_key_axis = torch.ones_like(self.outputs[0].max_key_axis)
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(self.outputs[0],
                                                                                                  QuantMode.to_symmetric(
                                                                                                      q_mode_activation),
                                                                                                  out.qbits,
                                                                                                  False)

        self.params["adjust_q"] = adjust_q
        if adjust_q >= 8:
            self.constants["log2_lut"] = PyTensor(self.name + "/log2_lut",
                                                  torch.tensor(table_Log2_q15).cpu().numpy().astype(
                                                      dtype2nptype(Dtype.INT16)))
            self.constants["log2_lut"].dtype = Dtype.INT16

            self.constants["pow2_lut"] = PyTensor(self.name + "/pow2_lut",
                                                  torch.tensor(table_pow2_q15).cpu().numpy().astype(
                                                      dtype2nptype(Dtype.UINT16)))
            self.constants["pow2_lut"].dtype = Dtype.UINT16
        self.params["quantize_method"] = 'FAST_EXP'
    else:
        out_sign = False or self.force_dtype_int
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(self.outputs[0],
                                                                                                  q_mode_activation,
                                                                                                  out.qbits,
                                                                                                  out_sign)
        self.params["quantize_method"] = 'LUT'
        mbits = max(16, 32 - ibits) if len(extra_params) < 2 else int(extra_params[1])
        max_val = torch.tensor((1 << mbits) - 1, device=dev)
        max_inp = torch.log(max_val)
        lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
        quant_range_linspace = torch.linspace(0, 2 ** inp.qbits - 1, steps=lsteps, device=dev)
        max_inp = max_inp - inp.zerop / inp.scale
        lut = linear_dequantize(quant_range_linspace - (2 ** inp.qbits - 1), inp.scale, inp.zerop) + max_inp
        lut = torch.exp(lut).round().clamp(0, max_val)
        self.constants["lut"] = PyTensor(self.name + "/explut",
                                         lut.cpu().numpy().astype(dtype2nptype(range2dtype(0, max_val.item())[1])))
        do_scale_bits = out.qbits
        ss = get_scale_approximation_params(out.scale, do_scale_bits, force_shift_positive=self.force_shift_positive)
        do_scale, do_scale_type, do_shift, do_shift_type = ss
        self.params["shift_value"] = int(do_shift)
        self.params["shift_type"] = do_shift_type
        self.params["scale_value"] = int(do_scale)
        self.params["scale_type"] = do_scale_type
        if Target.optimized_target_level(min_compatible_zhouyi_target) >= 2 and \
                axis == (len(inp.ir_shape) - 1) and inp.ir_shape[-1] <= 4096 and \
                not (h == 1 and w == 1 or len(shape) == 2):
            # Use X3 Softmax INT AIFF
            lut_items_in_bits = Target.aiff_lut_items_in_bits(min_compatible_zhouyi_target)
            self.params['is_perf_mode'] = True
            flut = 2 ** torch.linspace(0.0, 1.0, steps=2**lut_items_in_bits + 1, device=dev)
            flut = to_fp24(flut)
            self.constants["float_lut"] = PyTensor(
                self.name + "/fp24_lut", flut.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))


@approx_register(OpType.Softmax)
def softmax_approx(self, *args):
    approx_params = self.get_attrs('approx_params', optional=True, default_value=[0])
    method = int(approx_params[0] if len(approx_params) > 0 else 0)
    min_compatible_zhouyi_target = self.attrs["min_compatible_zhouyi_target"].upper()
    lut_items_in_bits = Target.aiff_lut_items_in_bits(min_compatible_zhouyi_target)
    if 1 == method and Target.optimized_target_level(min_compatible_zhouyi_target) >= 2:
        inp = self.inputs[0]
        dev = inp.betensor.device
        lut = 2 ** torch.linspace(0.0, 1.0, steps=2**lut_items_in_bits + 1, device=dev)
        lut = to_fp24(lut)
        self.constants["lut"] = PyTensor(self.name + "/fp24_lut", lut.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
        self.params['is_perf_mode'] = True  # use fast approximate implementation of AIFF as much as possible
        self.params['lut_mode'] = 'EXP'
    else:
        # not suit for aiff, need use tpc to implement a higher accuracy version
        self.params['is_perf_mode'] = False


@op_register(OpType.Softmax)
def softmax(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    axis = self.get_param('axis')
    shape_value_in_axis = inp.betensor.shape[axis]
    in_size = inp.betensor.numel() / shape_value_in_axis
    if self.quantized:
        if self.get_param('is_perf_mode', optional=True, default_value=False) and self.get_param('quantize_method', optional=True, default_value="") == 'LUT' and 'float_lut' in self.constants:
            # X3 AIFF INT Forward
            vx = inp.betensor.float()
            vx = to_fp24(vx)
            max_v, _ = vx.max(axis, keepdim=True)
            tmp = to_fp24(construct_torch_tensor(-1 / (0.6931471 * inp.scale),
                          device=inp.betensor.device))  # 0.69 = log(2)
            f_vdata = to_fp24((max_v - vx) * tmp)

            pow2_f_lut = self.constants["float_lut"].betensor.float()
            yy = x3_aiff_exp_approximation(f_vdata, pow2_f_lut)
            y_sum = yy.sum(axis, keepdim=True)
            y_sum = to_fp24(y_sum)

            score = to_fp24(1 / y_sum)

            yy16 = yy.half()
            yy24 = to_fp24(yy16.float())

            yy24 = yy24 * out.scale
            score = yy24 * score
            score = torch.round(score)
            score -= out.zerop
            score = torch.clamp(score, out.qmin, out.qmax)
            out.betensor = score.reshape(vx.shape)
            return out.betensor
        adjust_q = self.get_param('adjust_q', optional=True, default_value=-1)
        new_alg = adjust_q >= 0
        x = inp.betensor
        max_v, _ = x.max(axis, keepdim=True)
        if new_alg:
            if 1 < adjust_q < 8:
                OPT_FATAL(f"adjust_q should be in [0,1,8,9,10,11,"
                          f"12,13,14,15]")
            input_scale_value = self.get_ir_field(['input_scale', 'input_scale_value'], default_value=3)
            input_shift_value = self.get_ir_field(['input_shift', 'input_shift_value'], default_value=8)
            Pow2OutQ = self.params["pow2_out_q"]
            log2e_x = (x - max_v).long()

            if adjust_q >= 8:
                if (input_shift_value - adjust_q) > 0:
                    log2e_x = (log2e_x * input_scale_value) >> (input_shift_value - adjust_q)
                else:
                    log2e_x = (log2e_x * input_scale_value) << (adjust_q - input_shift_value)
                y = Pow2(log2e_x, adjust_q, Pow2OutQ)
            else:
                log2e_x = (log2e_x * input_scale_value) >> (input_shift_value)
                # *((Pow2OutQ+log2e_x)>=0) to fix torch version diff issue,lib neednot it
                y = (1 << (Pow2OutQ + log2e_x)) * ((Pow2OutQ + log2e_x) >= 0)
            y_sum = y.sum(axis, keepdim=True, dtype=torch.long)
        else:
            lut = self.constants["lut"].betensor
            x = x - max_v + 2 ** inp.qbits - 1
            x = torch.reshape(x, (-1,))
            y = lookup_lut_powerof2(x,
                                    lut,
                                    inp.qbits,
                                    False,
                                    dtype2bits(self.constants["lut"].dtype),
                                    is_signed(self.constants["lut"].dtype))
            y = torch.reshape(y, inp.betensor.shape)

            do_shift = self.params["shift_value"]
            do_scale = self.params["scale_value"]
            y_sum = y.sum(axis, keepdim=True, dtype=torch.long)

        '''
        now we support two kind of softmax quantized forward, the one shape_value_in_axis < 8 and in_size % 128 == 0 case
        existed is because of in small data numbers the saved enlarge_bits acurracy in division is faster than 31bits, the lib
        has better preformance, so in order to compatiblity the lib impl, we also has two kind of quantized forward.
        '''
        if new_alg:
            yy_sum = torch.clamp(y_sum, 1, (1 << 31) - 1)
            if adjust_q >= 8:
                log2_sum = Log2_norm(yy_sum, Pow2OutQ, adjust_q)
                yy_div = (log2e_x - log2_sum)
                out.betensor = Pow2(yy_div, adjust_q, out.qbits)
            else:
                log2_sum = simple_log2(yy_sum, Pow2OutQ, adjust_q)
                yy_div_sum = (((log2e_x+out.qbits) << adjust_q) - log2_sum) >> adjust_q
                # *(yy_div_sum>=0) to fix torch version diff issue,lib neednot it
                y_div_sum = (1 << yy_div_sum)*(yy_div_sum >= 0)
                out.betensor = linear_requantize(y_div_sum, 1.0, 0, 0, out.qmin, out.qmax)

        else:
            # 31->30: Adaptation the lut.max=1 case in android and these case lib would overflow using base_bits=31
            base_bits = 30 if lut.max() == 1 else 31
            if shape_value_in_axis < 8 and in_size % 128 == 0 and self.inputs[0].qbits <= 8:
                scale_bits = torch.ceil(torch.log2(lut.max())).long()
                enlarge_bits = base_bits - scale_bits
                numerator = (y.long() << enlarge_bits) + (y_sum >> 1)
                denominator = torch.maximum(y_sum, torch.ones_like(y_sum))
                z = torch.div(numerator, denominator, rounding_mode='trunc')
                do_shift = do_shift + enlarge_bits
                out.betensor = linear_requantize(z, do_scale, do_shift, out.zerop, out.qmin, out.qmax)
            else:
                denominator = torch.maximum(y_sum, torch.ones_like(y_sum))
                scale_bits = torch.ceil(torch.log2(torch.tensor(do_scale))).long()
                enlarge_bits = base_bits - scale_bits
                enlarge_shift = do_shift + enlarge_bits
                enlarge_scale = do_scale * (2 ** enlarge_bits)
                reverse_sum = torch.div(enlarge_scale, denominator, rounding_mode='trunc')
                y_div_sum = y.long() * reverse_sum
                out.betensor = linear_requantize(y_div_sum, 1.0, enlarge_shift, out.zerop, out.qmin, out.qmax)
    else:
        if self.approximated and "lut" in self.constants:
            lut = self.constants["lut"].betensor
            out.betensor = x3_aiff_softmax_approximation(inp.betensor, axis, lut)
        else:
            out.betensor = torch.nn.functional.softmax(inp.betensor.float(), dim=axis)
    return out.betensor
