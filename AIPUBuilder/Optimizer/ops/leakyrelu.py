# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils.quant_tool_utils import *
from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.framework import *

import torch


register_optype('LeakyRELU')


@op_register(OpType.LeakyRELU)
def leakyrelu(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]

    negative_slope = self.get_param("negative_slope_value")
    if self.quantized:
        #Yf = leakyrelu(Xf)
        # (Yq+Zy)/Sy = leakyrelu((Xq+Zx)/Sx)
        #Yq = leakyrelu(Xq+Zx) * Sy/Sx - Zy
        negative_slope_shift = self.get_param("negative_slope_shift", optional=True, default_value=out.qbits)
        zeros = torch.zeros_like(inp.betensor)
        x = inp.betensor + inp.zerop
        # here, need not clamp, to use linear_requantize, we using a bigest value in hardware platform
        tmp_qmin, tmp_qmax = bits2range(48, True)
        y = linear_requantize(torch.minimum(x, zeros), negative_slope, negative_slope_shift, -
                              torch.maximum(x, zeros), tmp_qmin, tmp_qmax)
        # y = torch.round(torch.maximum(x, zeros) + ((negative_slope * torch.minimum(x, zeros)*(0.5 **negative_slope_shift)))).int()
        if 'shift_value' in self.params:
            do_shift = self.params["shift_value"]
            do_scale = self.params["scale_value"]
        elif 'shift' in self.constants:
            do_shift = self.constants["shift"].betensor
            do_scale = self.constants["scale"].betensor
        else:
            do_shift = 0
            do_scale = 1

        out.betensor = linear_requantize(y, do_scale, do_shift, out.zerop, out.qmin, out.qmax)
    else:
        # out.betensor = torch.maximum(inp.betensor, zeros) + negative_slope * torch.minimum(inp.betensor, zeros)
        out.betensor = torch.nn.functional.leaky_relu(inp.betensor, negative_slope)
    return out.betensor


@quant_register(OpType.LeakyRELU)
def leakyrelu_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]

    q_mode_activation = self.attrs["q_mode_activation"]
    q_bits_activation = self.attrs["q_bits_activation"]
    multiplier_bits = self.attrs['multiplier_bits']
    out.qinvariant = False
    out.qbits = q_bits_activation
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, True)
    do_scale, do_scale_type, do_shift, do_shift_type = \
        get_scale_approximation_params(out.scale / inp.scale, mult_bits=multiplier_bits,
                                       force_shift_positive=self.force_shift_positive)

    negative_slope = int(round(self.get_param('negative_slope_value') * 2 ** out.qbits))
    self.params['negative_slope_value'] = negative_slope
    self.params['negative_slope_type'] = range2dtype(
        0, negative_slope)[1] if negative_slope > 0 else range2dtype(negative_slope, 0)[1]

    doscale_name = 'scale' if is_torch_tensor_with_multi_data(do_scale) else 'scale_value'
    doshift_name = 'shift' if is_torch_tensor_with_multi_data(do_shift) else 'shift_value'
    self.set_ir_field(doscale_name, do_scale, do_scale_type)
    self.set_ir_field(doshift_name, do_shift, do_shift_type)
    if not is_torch_tensor_with_multi_data(do_scale):
        self.params["shift_type"] = do_shift_type
        self.params["scale_type"] = do_scale_type
