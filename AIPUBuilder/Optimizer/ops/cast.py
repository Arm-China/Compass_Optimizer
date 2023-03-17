# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *


@op_register(OpType.Cast)
def cast(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    if self.quantized:
        do_scale = 1
        do_shift = 0
        if 'scale_value' in self.params:
            do_scale = self.get_param('scale_value')
            do_shift = self.get_param('shift_value')
        out.betensor = linear_requantize(inp.betensor + inp.zerop, do_scale, do_shift, out.zerop, out.qmin, out.qmax)
    else:
        if 'only_for_quantized' in self.params:
            out.betensor = inp.betensor
        elif is_float(self.params['to_dtype']):
            out.betensor = torch._cast_Float(inp.betensor)
        else:
            qmin, qmax = dtype2range(self.params['to_dtype'])
            qbits = dtype2bits(self.params['to_dtype'])
            neg_shift = (2**qbits)*torch.ones_like(inp.betensor.long())
            inp_tensor = inp.betensor.long()
            while torch.max(inp_tensor) > qmax or torch.min(inp_tensor) < qmin:
                inp_tensor = torch.where(inp_tensor < qmin, inp_tensor+neg_shift, inp_tensor)
                inp_tensor = torch.where(inp_tensor > qmax, inp_tensor-neg_shift, inp_tensor)
            out.betensor = inp_tensor.type(dtype2torch_type(self.params['to_dtype']))

    return out.betensor


@quant_register(OpType.Cast)
def cast_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")

    inp = self.inputs[0]
    out = self.outputs[0]
    if 'only_for_quantized' in self.params:
        if inp.qinvariant or inp.dtype == self.params['to_dtype']:
            out.scale = inp.scale
            out.zerop = inp.zerop
            out.dtype = self.params['to_dtype']
            out.qbits = dtype2bits(out.dtype)
            out.qinvariant = inp.qinvariant
        elif (is_float(self.params['to_dtype']) and is_float(inp.dtype)):
            out.scale = inp.scale
            out.zerop = inp.zerop
            out.dtype = self.params['to_dtype']
            out.qbits = inp.qbits
            out.qinvariant = inp.qinvariant
        else:
            if is_float(self.params['to_dtype']):
                OPT_FATAL("wrong to_dtype for only_for_quantized situation.")
            out.dtype = self.params['to_dtype']
            out.qbits = dtype2bits(out.dtype)
            out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
                out, q_mode_activation, out.qbits, is_signed(out.dtype))
            do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
                out.scale / inp.scale, mult_bits=out.qbits, force_shift_positive=self.force_shift_positive)
            self.params['scale_value'] = int(do_scale)
            self.params['scale_type'] = do_scale_type
            self.params['shift_value'] = int(do_shift)
            self.params['shift_type'] = do_shift_type
            out.qinvariant = False
        self.params.pop('only_for_quantized')
    else:
        out.scale = inp.scale
        out.zerop = inp.zerop
        out.qbits = inp.qbits
        out.dtype = inp.dtype
        out.qinvariant = inp.qinvariant
        self.params['to_dtype'] = out.dtype
