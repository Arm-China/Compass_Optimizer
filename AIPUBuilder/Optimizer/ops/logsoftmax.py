# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch


@quant_register(OpType.LogSoftmax)
def logsoftmax_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Softmax currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]
    min_compatible_zhouyi_target = self.attrs["min_compatible_zhouyi_target"].upper()

    inp = self.inputs[0]
    dev = inp.betensor.device
    if inp.qbits > 16:
        OPT_WARN(self.name + " : input tensor use %d bit quantization, which may cause softmax's lut table very large" % inp.qbits)

    if Target.optimized_target_level(min_compatible_zhouyi_target) <= 0:
        # softmax part
        max_val = torch.tensor(1 << (32 - 12), dtype=torch.float,
                               device=dev)  # so we allow to accumulate less than 2**12 items with a 32bit accumulator
        max_inp = linear_quantize_clip(torch.log(max_val), inp.scale, inp.zerop, torch.iinfo(torch.int64).min,
                                       torch.iinfo(torch.int64).max)

        lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
        lut = linear_dequantize(torch.linspace(0, 2 ** inp.qbits - 1, steps=lsteps, device=dev) + max_inp - 2 ** inp.qbits,
                                inp.scale, inp.zerop)
        lut = torch.exp(lut).round().clamp(0.0, max_val)
        self.constants["lut_exp"] = PyTensor(
            self.name + "/explut", lut.cpu().numpy().astype(dtype2nptype(Dtype.UINT32)))
        plh0 = self.placeholders[0]
        plh0.qbits = q_bits_activation
        plh0.qinvariant = False
        plh0.scale, plh0.zerop, plh0.qmin, plh0.qmax, plh0.dtype = get_linear_quant_params_from_tensor(
            plh0, q_mode_activation, q_bits_activation, False)
        do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
            plh0.scale, q_bits_activation, force_shift_positive=self.force_shift_positive)
        self.params["shift_value"] = int(do_shift)
        self.params["shift_type"] = do_shift_type
        self.params["scale_value"] = int(do_scale)
        self.params["scale_type"] = do_scale_type

        # log part
        q_mode_activation = self.attrs["q_mode_activation"]
        if QuantMode.is_per_channel(q_mode_activation) == True:
            OPT_FATAL("Currently not support per-channel quantization")
        q_bits_activation = self.attrs["q_bits_activation"]
        out = self.outputs[0]
        plh1 = self.placeholders[1]

        plh1.qbits = q_bits_activation
        plh1.scale, plh1.zerop, plh1.qmin, plh1.qmax, plh1.dtype = get_linear_quant_params_from_tensor(
            plh1, q_mode_activation, plh1.qbits, True)
        plh1.qinvariant = False

        out.qbits = plh1.qbits
        out.scale = plh1.scale
        out.zerop = plh1.zerop
        out.qmin = plh1.qmin
        out.qmax = plh1.qmax
        out.dtype = plh1.dtype
        out.qinvariant = plh1.qinvariant
        lut = torch.log(torch.linspace(0, 1, steps=lsteps, device=dev))
        lut = linear_quantize_clip(lut, out.scale, out.zerop, out.qmin, out.qmax)
        lut[0] = lut[1]
        self.constants["lut_log"] = PyTensor(self.name+"/log_lut", lut.cpu().numpy().astype(dtype2nptype(out.dtype)))
        self.constants["lut_log"].qbits = out.qbits
    else:
        lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
        val = torch.linspace(0, 2 ** inp.qbits - 1, steps=lsteps, device=dev).float()
        lut = linear_dequantize(val, -inp.scale, 0)
        lut = torch.exp(lut).flip(dims=[0])
        self.constants["lut_exp"] = PyTensor(self.name + "/explut",
                                             lut.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))

        out = self.outputs[0]
        out.qbits = inp.qbits
        out.min, out.max = -16, 0
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, out.qbits, True)
        self.params['quantize_scale'] = [float(out.scale / inp.scale), float(out.scale)]
        out.qmin, out.qmax = bits2range(out.qbits, True)
        out.dtype = bits2dtype(out.qbits, True)
        out.qinvariant = False
        # To be consistent with all fixed point flow parameters
        self.params["shift_value"] = int(0)
        self.params["shift_type"] = Dtype.INT8
        self.params["scale_value"] = int(1)
        self.params["scale_type"] = bits2dtype(q_bits_activation, False)


@op_register(OpType.LogSoftmax)
def logsoftmax(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    axis = self.get_param('axis')
    shape_value_in_axis = inp.betensor.shape[axis]
    in_size = inp.betensor.numel() / shape_value_in_axis
    if self.quantized:
        lut_dtype = self.constants["lut_exp"].dtype
        if lut_dtype == Dtype.FP32:
            x = inp.betensor.long()
            lut = self.constants["lut_exp"].betensor
            max_v, _ = x.max(axis, keepdim=True)
            _, max_value = bits2range(inp.qbits, False)

            index = max_value - max_v + x
            index_flatten = torch.reshape(index, (-1,))
            y = lookup_lut_powerof2(index_flatten, lut, inp.qbits, False, dtype2bits(self.constants["lut_exp"].dtype),
                                    is_signed(self.constants["lut_exp"].dtype), is_float=True)

            exp_value = torch.reshape(y, index.shape)
            sum_exp = exp_value.sum(axis, keepdim=True).float()
            log_sum_exp = torch.log(sum_exp)
            scale0, scale1 = self.params['quantize_scale']
            logsoftmax = (x - max_v).float() * scale0 - log_sum_exp * scale1
            out.betensor = torch.clamp(torch.round(logsoftmax).long() - out.zerop, out.qmin, out.qmax)
        else:
            x = inp.betensor
            lut = self.constants["lut_exp"].betensor
            max_v, _ = x.max(axis, keepdim=True)
            x = x - max_v + 2 ** inp.qbits - 1
            x = torch.reshape(x, (-1,))
            y = lookup_lut_powerof2(x, lut, inp.qbits, False, dtype2bits(self.constants["lut_exp"].dtype),
                                    is_signed(self.constants["lut_exp"].dtype))
            y = torch.reshape(y, inp.betensor.shape)

            do_shift = self.params["shift_value"]
            do_scale = self.params["scale_value"]
            y_sum = y.sum(axis, keepdim=True).long()
            if shape_value_in_axis < 8 and in_size % 128 == 0:
                # y is 20bit
                z = torch.div(((y.long() << 11) + (y_sum >> 1)),
                              (torch.maximum(y_sum, torch.ones_like(y_sum))), rounding_mode='trunc')
                do_shift = do_shift + 11

                # z = z * do_scale
                # if do_shift < 0:
                #     z = z << (-1 * do_shift)
                # else:
                #     z = z >> do_shift
                # x2 = torch.clamp(z.float(), 0, out.qmax - out.qmin)

                x2 = linear_requantize(z, do_scale, do_shift, 0, 0, out.qmax - out.qmin)
            else:
                # y is 20bit
                # y * reverse_sum <= 2 ** 31
                # 31->30: Adaptation the lut.max=1 case in android and these case lib would overflow using base_bits=31
                base_bits = 30 if lut.max() == 1 else 31
                denominator = torch.maximum(y_sum, torch.ones_like(y_sum))
                scale_bits = torch.ceil(torch.log2(torch.tensor(do_scale))).long()
                enlarge_bits = base_bits - scale_bits
                enlarge_shift = do_shift + enlarge_bits
                enlarge_scale = do_scale * (2 ** enlarge_bits)
                reverse_sum = torch.div(enlarge_scale, denominator, rounding_mode='trunc')
                y_div_sum = y.long() * reverse_sum
                x2 = linear_requantize(y_div_sum, 1.0, enlarge_shift, 0, 0, out.qmax - out.qmin)

            x2[x2 != x2] = 0  # nan_to_num

            log_lut = self.constants["lut_log"]
            x2 = torch.reshape(x2, (-1,))
            # y2 = torch.gather(log_lut, 0, x2.long())
            y2 = lookup_lut_powerof2(x2, log_lut.betensor, inp.qbits, False,
                                     dtype2bits(log_lut.dtype), is_signed(log_lut.dtype))
            y2 = torch.reshape(y2, inp.betensor.shape)
            out.betensor = y2
    else:
        out.betensor = torch.log_softmax(inp.betensor, dim=axis)
        softmax_func = torch.nn.Softmax(dim=axis)
        softmax_output = softmax_func(inp.betensor)
        placeholders0_tensor = torch.clamp(softmax_output, OPT_EPSILON, 1)
        placeholders1_tensor = torch.log(placeholders0_tensor)
        if len(self.placeholders) < 2:
            ph0 = PyTensor(self.name+"/softmax_outputs",
                           placeholders0_tensor.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            ph1 = PyTensor(self.name+"/log_outputs",
                           placeholders1_tensor.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            self.placeholders.append(ph0)
            self.placeholders.append(ph1)
        self.placeholders[0].betensor = placeholders0_tensor
        self.placeholders[1].betensor = placeholders1_tensor
    return out.betensor
