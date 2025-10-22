# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import OPT_WARN
from AIPUBuilder.Optimizer.utils import *
import torch


@op_register(OpType.Mod)
def mod(self, *args):
    dividend, divisor = self.inputs[0], self.inputs[1]
    fmod = self.get_param("fmod")

    '''if want to use np.fmod or np.mod:'''
    # np_dividend = dividend.to_numpy()
    # np_divisor = divisor.to_numpy()
    # import numpy as np
    # _func_map = {
    #         True: np.fmod,
    #         False: np.mod
    #     }
    def torch_mod(a, b):
        # return a - a.div(b, rounding_mode='floor') * b
        try:
            ret = a - a.div(b, rounding_mode='floor') * b
        except Exception as e:
            OPT_WARN(f"{e} in mod op.")
            b[b == 0] = 1
            ret = a - a.div(b, rounding_mode='floor') * b
        return ret

    def torch_fmod(a, b):
        try:
            ret = torch.fmod(a, b)
        except Exception as e:
            OPT_WARN(f"{e} in mod op.")
            b[b == 0] = 1
            ret = torch.fmod(a, b)
        return ret

    _func_map = {
        True: torch_fmod,
        False: torch_mod,
    }

    if self.quantized:
        dividend_scale, divisor_scale = self.get_param('scale_value')
        dividend_shift, divisor_shift = self.get_param('shift_value')
        de_inpq0 = linear_requantize(dividend.betensor + dividend.zerop,
                                     dividend_scale,
                                     dividend_shift,
                                     0,
                                     dividend.qmin,
                                     dividend.qmax)
        de_inpq1 = linear_requantize(divisor.betensor + divisor.zerop,
                                     divisor_scale,
                                     divisor_shift,
                                     0,
                                     divisor.qmin,
                                     divisor.qmax)

        out = _func_map[fmod](de_inpq0, de_inpq1)
        out = torch.clamp(out - self.outputs[0].zerop,
                          self.outputs[0].qmin,
                          self.outputs[0].qmax)
    else:
        '''if we want to use np.fmod or np.mod:'''
        # out = _func_map[fmod](np_dividend, np_divisor)
        # out = torch.tensor(out, device=dividend.betensor.device)
        out = _func_map[fmod](dividend.betensor.float(), divisor.betensor.float())

    self.outputs[0].betensor = out
    return self.outputs[0].betensor


@quant_register(OpType.Mod)
def mod_quantize(self, *args):
    q_bits_activation = self.attrs["q_bits_activation"]
    q_mode_activation = self.attrs["q_mode_activation"]

    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]

    out_signed = is_signed(inp0.dtype)
    out.qbits = q_bits_activation
    out.qmin, out.qmax = bits2range(out.qbits, out_signed)
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, is_signed=out_signed)
    scale0, shift0 = 1, 0
    scale1, shift1 = 1, 0
    scale0_type, shift0_type = Dtype.UINT8, Dtype.INT8
    scale1_type, shift1_type = Dtype.UINT8, Dtype.INT8
    if inp0.qinvariant and inp1.qinvariant:
        out.qinvariant = True
        out.scale = 1.0
        out.zerop = 0
    elif inp0.qinvariant:
        scale1, scale1_type, shift1, shift1_type = get_scale_approximation_params(
            out.scale / inp1.scale, mult_bits=out.qbits, force_shift_positive=self.force_shift_positive)
        out.qinvariant = False
    elif inp1.qinvariant:
        scale0, scale0_type, shift0, shift0_type = get_scale_approximation_params(
            out.scale / inp0.scale, mult_bits=out.qbits, force_shift_positive=self.force_shift_positive)
        out.qinvariant = False
    else:
        scale0, scale0_type, shift0, shift0_type = get_scale_approximation_params(
            out.scale / inp0.scale, mult_bits=out.qbits, force_shift_positive=self.force_shift_positive)
        scale1, scale1_type, shift1, shift1_type = get_scale_approximation_params(
            out.scale / inp1.scale, mult_bits=out.qbits, force_shift_positive=self.force_shift_positive)
        out.qinvariant = False

    self.params["scale_value"] = [int(scale0), int(scale1)]
    self.params["scale_type"] = [scale0_type, scale1_type]
    self.params["shift_value"] = [int(shift0), int(shift1)]
    self.params["shift_type"] = [shift0_type, shift1_type]
