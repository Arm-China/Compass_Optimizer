# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
from AIPUBuilder.Optimizer.features import apply_calibration_strategy
import torch


@op_register(OpType.MatMul)
def matmul_forward(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]
    x = inp0.betensor.float()
    y = inp1.betensor.float()
    aasrb = self.get_param('remain_shift', optional=True, default_value=None)
    if self.get_param('trans_a'):
        if x.dim() == 0:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.transpose(-1, -2)
    if self.get_param('trans_b'):
        if y.dim() == 0:
            y = y.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 1:
            y = y.unsqueeze(0)
        y = y.transpose(-1, -2)
    if self.quantized:
        if is_torch_tensor_with_multi_data(inp0.zerop):
            inp0_zerop = inp0.broadcast_zerop if not self.get_param(
                'trans_a') else inp0.broadcast_zerop.transpose(-1, -2)
        else:
            inp0_zerop = inp0.zerop
        if is_torch_tensor_with_multi_data(inp1.zerop):
            inp1_zerop = inp1.broadcast_zerop if not self.get_param(
                'trans_b') else inp1.broadcast_zerop.transpose(-1, -2)
        else:
            inp1_zerop = inp1.zerop
        x = x + inp0_zerop
        y = y + inp1_zerop
    z = torch.matmul(x, y)
    if self.quantized:
        if all(sname in self.constants.keys() for sname in ['scale1', 'scale2', 'scale3']):
            # act per channel quantization
            a_scale = self.constants['scale1'].betensor
            b_scale = self.constants['scale2'].betensor
            c_scale = self.constants['scale3'].betensor
            z = z * a_scale * b_scale
            z = linear_quantize_clip(z.float(
            ), c_scale, 0, self.outputs[0].qmin, self.outputs[0].qmax, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', self.outputs[0].dtype))
        else:
            requant_shift = self.get_ir_field(['shift', 'shift_value'], default_value=0)
            requant_scale = self.get_ir_field(['scale', 'scale_value'], default_value=1)
            if aasrb is not None and not is_float(self.inputs[0].dtype) and not is_float(self.inputs[1].dtype) and (dtype2bits(self.inputs[1].dtype) > 8 or dtype2bits(self.inputs[0].dtype) > 8):
                z, requant_shift = aiff_ahead_shift_bias(z, requant_shift, None, int(aasrb))
            z = linear_requantize(z, requant_scale, requant_shift, out.broadcast_zerop, out.qmin,
                                  out.qmax, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', out.dtype))
    else:
        z *= self.get_param('fused_multiplier', optional=True, default_value=1)
    out.betensor = z
    return out.betensor


@quant_register(OpType.MatMul)
def matmul_quantize(self, *args):
    quant_type = self.attrs.get('quant_type')
    fpx_quantize = True if quant_type != 'disable' and QuantType.is_float(quant_type) else False
    q_mode_activation = self.attrs["q_mode_activation"]
    q_bits_activation = self.attrs["q_bits_activation"]
    multiplier_bits = self.get_attrs('multiplier_bits', optional=True, default_value=q_bits_activation)

    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]
    out_signed = is_signed(inp0.dtype) or is_signed(inp1.dtype)
    out.qinvariant = inp0.qinvariant and inp1.qinvariant
    if out.qinvariant:
        out.scale = 1.0
        out.zerop = 0
        out.qbits, out.dtype = range2dtype(out.extrema_min, out.extrema_max, force_int=self.force_dtype_int)
    else:
        out.qbits = q_bits_activation
        if fpx_quantize:
            out.dtype = QuantType.activation_type(quant_type)
            out.scale, out.zerop, out.qmin, out.qmax = get_fpx_quant_params_from_tensor(
                out, q_mode_activation, out.dtype)
        else:
            out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
                out, q_mode_activation, out.qbits, is_signed=out_signed)
    out_scale = out.scale
    if 'fused_multiplier' in self.params:
        fused_multiplier = self.params['fused_multiplier']
        self.params.pop('fused_multiplier')
        out_scale = out.scale * fused_multiplier
    if any(is_torch_tensor_with_multi_data(edge_scale) for edge_scale in [inp0.scale, inp1.scale, out_scale]):
        # act per channel quantization
        self.constants['scale1'] = PyTensor(
            'scale1', 1.0 / (inp0.broadcast_scale if not self.get_param('trans_a') else inp0.broadcast_scale.transpose(-1, -2)), dtype=Dtype.FP32)
        self.constants['scale2'] = PyTensor(
            'scale2', 1.0 / (inp1.broadcast_scale if not self.get_param('trans_b') else inp1.broadcast_scale.transpose(-1, -2)), dtype=Dtype.FP32)
        self.constants['scale3'] = PyTensor('scale3', out_scale.reshape(out.broadcast_scale.shape), dtype=Dtype.FP32)
    else:
        ts = out_scale / (inp0.scale * inp1.scale)
        if fpx_quantize:
            do_scale_type = Dtype.FP32
            do_scale = ts
            doscale_name = 'scale' if is_torch_tensor_with_multi_data(do_scale) else 'scale_value'
            self.set_ir_field(doscale_name, do_scale, do_scale_type)
            if not is_torch_tensor_with_multi_data(do_scale):
                self.params["scale_type"] = do_scale_type
        else:
            do_scale, do_scale_type, do_shift, do_shift_type = \
                get_scale_approximation_params(ts,
                                               multiplier_bits,
                                               force_shift_positive=self.force_shift_positive)
            if 'remain_shift' in self.attrs:
                self.params['remain_shift'] = self.attrs['remain_shift']
            doscale_name = 'scale' if is_torch_tensor_with_multi_data(do_scale) else 'scale_value'
            doshift_name = 'shift' if is_torch_tensor_with_multi_data(do_shift) else 'shift_value'
            self.set_ir_field(doscale_name, do_scale, do_scale_type)
            self.set_ir_field(doshift_name, do_shift, do_shift_type)
            if not is_torch_tensor_with_multi_data(do_scale):
                self.params["shift_type"] = do_shift_type
                self.params["scale_type"] = do_scale_type
