# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch


@quant_register(OpType.RELU)
def relu_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    q_bits_activation = self.attrs["q_bits_activation"]
    quant_type = self.attrs['quant_type']
    fp8_quantize = True if quant_type != 'disable' and QuantType.is_float(quant_type) else False

    inp = self.inputs[0]
    out = self.outputs[0]
    if fp8_quantize:
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
        out.qinvariant = inp.qinvariant

        do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
            out.scale / inp.scale, mult_bits=out.qbits, force_shift_positive=self.force_shift_positive)
        scale_name = 'scale' if is_torch_tensor_with_multi_data(do_scale) else 'scale_value'
        shift_name = 'shift' if is_torch_tensor_with_multi_data(do_shift) else 'shift_value'
        self.set_ir_field(scale_name, do_scale, do_scale_type)
        self.set_ir_field(shift_name, do_shift, do_shift_type)
        if not is_torch_tensor_with_multi_data(do_scale):
            self.params["shift_type"] = do_shift_type
            self.params["scale_type"] = do_scale_type


@op_register(OpType.RELU)
def relu(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    if self.quantized:
        #Yf = relu(Xf)
        # (Yq+Zy)/Sy = relu((Xq+Zx)/Sx)
        #Yq = relu(Xq+Zx) * Sy/Sx - Zy
        # on cpu device, torch.relu does not support 'clamp_min_cpu' for half, so use inp.betensor.float()
        y = torch.nn.functional.relu(inp.betensor.float() + inp.zerop)
        do_shift = self.get_ir_field(['shift_value', 'shift'], default_value=0)
        do_scale = self.get_ir_field(['scale_value', 'scale'], default_value=1)
        out.betensor = linear_requantize(y, do_scale, do_shift, out.zerop, out.qmin, out.qmax,
                                         out.key_axis, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', out.dtype))
    else:
        out.betensor = torch.nn.functional.relu(inp.betensor.float())
    return out.betensor
