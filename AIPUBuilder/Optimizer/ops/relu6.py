# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch


@quant_register(OpType.RELU6)
def relu6_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]
    quant_type = self.attrs['quant_type']
    fpx_quantize = True if quant_type != 'disable' and QuantType.is_float(quant_type) else False

    inp = self.inputs[0]
    out = self.outputs[0]
    if fpx_quantize:
        out.dtype = QuantType.activation_type(quant_type)
        out.qbits = dtype2bits(out.dtype)
        out.scale, out.zerop, out.qmin, out.qmax = get_fpx_quant_params_from_tensor(out, q_mode_activation, out.dtype)
        out.qinvariant = False

        do_scale_type = Dtype.FP32
        do_scale = out.scale / inp.scale
        scale_name = 'scale' if is_torch_tensor_with_multi_data(do_scale) else 'scale_value'
        self.set_ir_field(scale_name, do_scale, do_scale_type)
        if not is_torch_tensor_with_multi_data(do_scale):
            self.params["scale_type"] = do_scale_type
    else:
        out.qbits = q_bits_activation
        out_sign = False or self.force_dtype_int
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, out.qbits, out_sign)
        do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
            out.scale / inp.scale, mult_bits=out.qbits, force_shift_positive=self.force_shift_positive)
        self.params["shift_value"] = int(do_shift)
        self.params["shift_type"] = do_shift_type
        self.params["scale_value"] = int(do_scale)
        self.params["scale_type"] = do_scale_type
    out.qinvariant = inp.qinvariant


@op_register(OpType.RELU6)
def relu6(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    if self.quantized:
        #Yf = relu(Xf)
        # (Yq+Zy)/Sy = relu((Xq+Zx)/Sx)
        #Yq = relu(Xq+Zx) * Sy/Sx - Zy
        y = torch.nn.functional.relu(inp.betensor + inp.zerop)
        do_shift = 0
        do_scale = 1
        if "shift" not in self.constants:
            do_shift = self.params["shift_value"]
            do_scale = self.params["scale_value"]
        else:
            do_shift = self.constants["shift"].betensor
            do_scale = self.constants["scale"].betensor
        out.betensor = linear_requantize(y, do_scale, do_shift, out.zerop, out.qmin, out.qmax,
                                         round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', out.dtype))
    else:
        out.betensor = torch.nn.functional.relu6(inp.betensor)
    return out.betensor
