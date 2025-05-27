# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *


def forward_with_clip(inp, out_dtype, clip_mode, input_zerop=0, output_zerop=0):
    if is_float(out_dtype):
        output = torch._cast_Float(inp)
    else:
        # currently truncation doesn't need to consider inputzp and outputzp
        if clip_mode == 'TRUNCATION':
            inp_tensor = inp.cpu().numpy()
            out_array = inp_tensor.astype(dtype2nptype(out_dtype))
            output = PyTensor('tmp', out_array).betensor.to(inp.device)
        else:
            qmin, qmax = dtype2range(out_dtype)
            inp_t = inp.long() + input_zerop - output_zerop
            output = torch.clamp(inp_t, qmin, qmax)

    return output


@op_register(OpType.Cast)
def cast(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    clip_mode = self.get_param('clip_mode', optional=True, default_value='saturation').upper()
    ignore_scale_zp = self.get_param('ignore_scale_zp', optional=True, default_value=False)
    if self.quantized:
        if self.get_ir_field(['scale_value', 'scale']) is not None:
            do_scale = self.get_ir_field(['scale_value', 'scale'])
            do_shift = self.get_ir_field(['shift_value', 'shift'])
            out.betensor = linear_requantize(inp.betensor + inp.broadcast_zerop, do_scale, do_shift, out.zerop,
                                             out.qmin, out.qmax, out.key_axis)
        else:
            input_zerop = 0 if ignore_scale_zp else inp.zerop
            output_zerop = 0 if ignore_scale_zp else out.zerop
            out.betensor = forward_with_clip(inp.betensor, self.params['to_dtype'],
                                             clip_mode, input_zerop, output_zerop)
    else:
        if 'only_for_quantized' in self.params:
            out.betensor = inp.betensor
        else:
            out.betensor = forward_with_clip(inp.betensor, self.params['to_dtype'], clip_mode)
    return out.betensor


@quant_register(OpType.Cast)
def cast_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    ignore_scale_zp = self.get_param('ignore_scale_zp', optional=True, default_value=True)
    inp = self.inputs[0]
    out = self.outputs[0]
    if 'only_for_quantized' in self.params:
        if inp.qinvariant or inp.dtype == self.params['to_dtype']:
            out.scale = inp.scale
            out.zerop = inp.zerop
            out.dtype = self.params['to_dtype']
            out.qbits = dtype2bits(out.dtype)
            out.qinvariant = inp.qinvariant
            self.params['ignore_scale_zp'] = True
        elif (is_float(self.params['to_dtype']) and is_float(inp.dtype)):
            out.scale = inp.scale
            out.zerop = inp.zerop
            out.dtype = self.params['to_dtype']
            out.qbits = inp.qbits
            out.qinvariant = inp.qinvariant
            self.params['ignore_scale_zp'] = True
        else:
            if is_float(self.params['to_dtype']):
                OPT_FATAL("wrong to_dtype for only_for_quantized situation.")
            out.dtype = self.params['to_dtype']
            out.qbits = dtype2bits(out.dtype)

            if out.qbits > 8:
                q_mode_activation = QuantMode.to_symmetric(q_mode_activation)
            elif is_signed(inp.dtype) and not is_signed(out.dtype):
                q_mode_activation = QuantMode.to_asymmetric(q_mode_activation)
            out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
                out, q_mode_activation, out.qbits, is_signed(out.dtype))
            do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
                out.scale / inp.scale, mult_bits=out.qbits, force_shift_positive=self.force_shift_positive)

            scale_name = "scale" if is_torch_tensor_with_multi_data(do_scale) else "scale_value"
            shift_name = "shift" if is_torch_tensor_with_multi_data(do_shift) else "shift_value"
            self.set_ir_field(scale_name, do_scale, do_scale_type)
            self.set_ir_field(shift_name, do_shift, do_shift_type)
            if not is_torch_tensor_with_multi_data(do_scale):
                self.params["shift_type"] = do_shift_type
                self.params["scale_type"] = do_scale_type
            self.params['ignore_scale_zp'] = False
            out.qinvariant = False
        self.params['clip_mode'] = 'saturation'
        self.params.pop('only_for_quantized')
    else:
        if not is_float(inp.ir_dtype):
            out.scale = inp.scale
            out.zerop = inp.zerop
            out.dtype = inp.dtype
            out.qbits = inp.qbits
            self.params['to_dtype'] = inp.dtype
            out.qinvariant = inp.qinvariant
        else:
            # Currently float->int will perform dequantization to int, which may have a little precision problems
            is_sign = is_signed(self.params['to_dtype'])
            out.qbits = min(32, dtype2bits(self.params['to_dtype']))
            out.dtype = bits2dtype(out.qbits, is_sign)
            do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
                1.0 / inp.scale, mult_bits=16, force_shift_positive=self.force_shift_positive)
            scale_name = "scale" if is_torch_tensor_with_multi_data(do_scale) else "scale_value"
            shift_name = "shift" if is_torch_tensor_with_multi_data(do_shift) else "shift_value"
            self.set_ir_field(scale_name, do_scale, do_scale_type)
            self.set_ir_field(shift_name, do_shift, do_shift_type)
            if not is_torch_tensor_with_multi_data(do_scale):
                self.params["shift_type"] = do_shift_type
                self.params["scale_type"] = do_scale_type
            out.scale = 1.0
            out.zerop = 0
            out.qinvariant = True
            self.params['ignore_scale_zp'] = (inp.scale.max() - 1.0).abs().item() <= torch.finfo(
                torch.float32).eps and (inp.scale.min() - 1.0).abs().item() <= torch.finfo(torch.float32).eps
            self.params['clip_mode'] = 'saturation'.upper()
