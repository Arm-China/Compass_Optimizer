# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch


@quant_register(OpType.RELU)
def relu_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    out = self.outputs[0]
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
        y = torch.nn.functional.relu(inp.betensor.float() + inp.zerop)
        do_shift = 0
        do_scale = 1
        do_shift = self.get_ir_field(['shift_value', 'shift'])
        do_scale = self.get_ir_field(['scale_value', 'scale'])
        out.betensor = linear_requantize(y, do_scale, do_shift, out.zerop, out.qmin, out.qmax, out.key_axis)
    else:
        out.betensor = torch.nn.functional.relu(inp.betensor)
    return out.betensor
