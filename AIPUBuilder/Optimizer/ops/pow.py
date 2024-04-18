# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch.nn as nn


def is_int(tensor):
    int_tensor = torch.round(tensor.float())
    diff_tensor = torch.abs(int_tensor - tensor)
    if diff_tensor.max() < torch.finfo(torch.float32).eps:
        return True
    return False


@op_register(OpType.Pow)
def pow(self, *args):
    base = self.inputs[0].betensor
    # in most cases, power is integer (scale == 1, zerop == 0)
    power = self.inputs[1].betensor

    if not self.quantized:
        inputs = torch.abs(base) + torch.finfo(torch.float32).eps
        log_outputs = torch.log(inputs)
        power_outputs = log_outputs * power
        exp_outputs = torch.exp(power_outputs)
        if len(self.placeholders) < 1:
            ph0 = PyTensor(self.name+"/log_outputs",
                           log_outputs.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            ph1 = PyTensor(self.name+"/power_outputs",
                           power_outputs.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            ph2 = PyTensor(self.name+"/exp_outputs",
                           exp_outputs.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            self.placeholders.append(ph0)
            self.placeholders.append(ph1)
            self.placeholders.append(ph2)
        self.placeholders[0].betensor = log_outputs
        self.placeholders[1].betensor = power_outputs
        self.placeholders[2].betensor = exp_outputs

        outputs = torch.pow(base, power)
    else:
        power += int(self.inputs[1].zerop)
        lut_log = self.constants['lut_log']
        shift = self.params["shift_value"]
        scale = self.params["scale_value"]

        x = torch.reshape(base, (-1,))
        ln_out = lookup_lut_powerof2(x, lut_log.betensor, self.inputs[0].qbits, is_signed(
            self.inputs[0].dtype), dtype2bits(lut_log.dtype), is_signed(lut_log.dtype))
        if "exponent" in self.params:
            ln_out = torch.reshape(ln_out, base.shape)
            outputs = ln_out
        else:
            lut_exp = self.constants['lut_exp']
            ln_out = torch.reshape(ln_out, base.shape)
            power_output_q = ln_out * power
            # requantize to ph1 (placeholder, symetric quantized to int16)
            pdtype = Dtype.INT16
            pmin, pmax = dtype2range(pdtype)
            pbits = dtype2bits(pdtype)
            power_output_q = linear_requantize(
                power_output_q, scale, shift, 0, pmin, pmax)

            power_output_q = torch.reshape(power_output_q, (-1,))
            exp_out = lookup_lut_powerof2(power_output_q, lut_exp.betensor, pbits, is_signed(
                pdtype), dtype2bits(lut_exp.dtype), is_signed(lut_exp.dtype))
            outputs = torch.reshape(exp_out, base.shape)

            inp_sign = torch.sign(base + self.inputs[0].zerop)
            # sign_power = (power + self.inputs[1].zerop) / self.inputs[1].scale
            # only when scale = 1.0, zerop = 0, sign_power is always int, then pow(-1, N) is meaningful
            sign_power = power
            out_sign = torch.pow(inp_sign, sign_power)
            outputs = outputs * out_sign
    self.outputs[0].betensor = outputs
    return outputs


def lut_exp_establish(inp, out, lsteps):
    input_range = torch.linspace(
        inp.qmin, inp.qmax, steps=lsteps, device=inp.betensor.device)
    xs = linear_dequantize(input_range, inp.scale, inp.zerop)
    lut = torch.exp(xs)
    lut = linear_quantize_clip(lut, out.scale, out.zerop, out.qmin, out.qmax)
    return lut


def lut_log_abs_establish(inp, out, lsteps):
    input_range = torch.linspace(
        inp.qmin, inp.qmax, steps=lsteps, device=inp.betensor.device)
    xs = linear_dequantize(input_range, inp.scale, inp.zerop)
    lut = torch.log(torch.abs(xs))
    lut = linear_quantize_clip(lut, out.scale, out.zerop, out.qmin, out.qmax)
    return lut


@quant_register(OpType.Pow)
def pow_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_activation = self.attrs["q_bits_activation"]

    if (self.inputs[1].scale != 1) and (is_signed(self.inputs[0].dtype)):
        OPT_WARN('layer_id=%s, type=%s, the base is signed and the exponent is float, making the result unpredictable' % (
            self.attrs['layer_id'], str(self.type)))

    out = self.outputs[0]
    inp = self.inputs[0]
    inp_signed = is_signed(inp.dtype)
    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    plh0_dtype = Dtype.INT16
    plh1_dtype = Dtype.INT16

    if "exponent" in self.params:
        power = self.get_param("exponent")
        out_signed = inp_signed and (power % 2 != 0)  # **2 always unsigned
        out.qbits = q_bits_activation
        out.qinvariant = inp.qinvariant and (0.0 == power - float(int(power)))
        if out.qinvariant:
            out.scale = 1.0
            out.zerop = 0
            out.dtype = bits2dtype(out.qbits, is_signed=out_signed)
            out.qmin, out.qmax = dtype2range(out.dtype)
        else:
            out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
                out, q_mode_activation, out.qbits, is_signed=out_signed)
        # make lut1 align to plh0&plh1, and lut2 counteracts the differences
        lut1 = linear_dequantize(torch.linspace(
            inp.qmin, inp.qmax, steps=lsteps, device=inp.betensor.device), inp.scale, inp.zerop)
        lut1 = torch.pow(lut1, power)
        lut1 = linear_quantize_clip(
            lut1, out.scale, out.zerop, out.qmin, out.qmax)
        pmin, pmax = dtype2range(plh1_dtype)
        lut2 = torch.linspace(
            pmin, pmax, steps=lsteps, device=inp.betensor.device).round().clamp(pmin, pmax)
        # if not is_signed(out.dtype) :
        #     lut1 -= 2 ** (out.qbits - 1)
        # lut1 = lut1.int() << (dtype2bits(plh1_dtype) - dtype2bits(out.dtype))
        # #lut2 = (lut2 / linear_quantize_clip(power, self.inputs[1].scale, self.inputs[1].zerop, self.inputs[1].qmin, self.inputs[1].qmax)).round()
        # lut2 = lut2.int() >> (dtype2bits(plh1_dtype) - dtype2bits(out.dtype))
        # if not is_signed(out.dtype) :
        #     lut2 += 2 ** (out.qbits - 1)
        self.constants["lut_log"] = PyTensor(self.name+"/lut_log", lut1.cpu().numpy().astype(dtype2nptype(out.dtype)))
        # self.constants["lut_exp"] = PyTensor(
        #     self.name+"/lut_exp", lut2.cpu().numpy().astype(dtype2nptype(out.dtype)))

        # to counteract power when calculate 'power * lut_log * do_scale >> do_shift'
        # rscale = 1.0 / linear_quantize_clip(power, self.inputs[1].scale, self.inputs[1].zerop, self.inputs[1].qmin, self.inputs[1].qmax).item()
        # do_scale, do_scale_type, do_shift, do_shift_type = \
        # get_scale_approximation_params(rscale, q_bits_activation, force_shift_positive=self.force_shift_positive)
        # self.params["shift_value"] = do_shift
        # self.params["shift_type"] = do_shift_type
        # self.params["scale_value"] = do_scale
        # self.params["scale_type"] = do_scale_type
        self.params["shift_value"] = 0
        self.params["shift_type"] = Dtype.INT8
        self.params["scale_value"] = 1
        self.params["scale_type"] = Dtype.UINT8
    else:
        power = self.inputs[1]
        scale_signed_list = [
            is_signed(plh0_dtype), is_signed(plh1_dtype), inp_signed]
        phd_qbits_list = [dtype2bits(plh0_dtype), dtype2bits(
            plh1_dtype), q_bits_activation]
        for idx, placeholders in enumerate(self.placeholders):
            placeholders.qbits = phd_qbits_list[idx]
            placeholders.scale, placeholders.zerop, placeholders.qmin, placeholders.qmax, placeholders.dtype = get_linear_quant_params_from_tensor(
                placeholders, QuantMode.to_symmetric(q_mode_activation), placeholders.qbits, is_signed=scale_signed_list[idx])
            placeholders.qinvariant = False
        plh0 = self.placeholders[0]
        plh1 = self.placeholders[1]
        plh2 = self.placeholders[2]
        plh2.qinvariant = self.inputs[0].qinvariant and self.inputs[1].qinvariant
        if plh2.qinvariant:
            plh2.scale = 1.0
            plh2.zerop = 0

        lut_names = ['lut_log', 'lut_exp']
        lut_log = lut_log_abs_establish(inp, plh0, lsteps)
        lut_exp = lut_exp_establish(plh1, plh2, lsteps)
        lut_list = [lut_log, lut_exp]
        lut_signed_list = [is_signed(plh0.dtype), is_signed(plh2.dtype)]
        lut_bits_list = [plh0.qbits, plh2.qbits]
        for idx, lut_name in enumerate(lut_names):
            self.constants[lut_name] = PyTensor(self.name+lut_name, lut_list[idx].cpu().numpy().astype(
                dtype2nptype(bits2dtype(lut_bits_list[idx], is_signed=lut_signed_list[idx]))))
            self.constants[lut_name].dtype = bits2dtype(
                lut_bits_list[idx], is_signed=lut_signed_list[idx])

        local_rescale = plh1.scale / (power.scale * plh0.scale)
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(local_rescale,
                                           mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)

        self.params["shift_value"] = int(do_shift)
        self.params["shift_type"] = do_shift_type
        self.params["scale_value"] = int(do_scale)
        self.params["scale_type"] = do_scale_type

        out.qbits = plh2.qbits
        out.scale = plh2.scale
        out.zerop = plh2.zerop
        out.qmin = plh2.qmin
        out.qmax = plh2.qmax
        out.dtype = plh2.dtype
        out.qinvariant = plh2.qinvariant
