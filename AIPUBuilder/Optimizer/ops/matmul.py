# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

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
        requant_shift = self.get_ir_field(['shift', 'shift_value'], default_value=0)
        requant_scale = self.get_ir_field(['scale', 'scale_value'], default_value=1)
        if aasrb is not None:
            z, requant_shift = aiff_ahead_shift_bias(z, requant_shift, None, int(aasrb))
        z = linear_requantize(z, requant_scale, requant_shift, out.broadcast_zerop, out.qmin, out.qmax)
    else:
        if len(self.placeholders) < 1:
            ph0 = PyTensor(self.name + "/matmul_result", z.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            self.placeholders.append(ph0)
        self.placeholders[0].betensor = z.clone().detach()
        self.placeholders[0].key_axis = self.outputs[0].key_axis
        z *= self.get_param('fused_multiplier', optional=True, default_value=1)
    out.betensor = z
    return out.betensor


@quant_register(OpType.MatMul)
def matmul_quantize(self, *args):
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
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, out.qbits, is_signed=out_signed)
    out_scale = out.scale
    if 'fused_multiplier' in self.params:
        fused_multiplier = self.params['fused_multiplier']
        self.params.pop('fused_multiplier')
        plh = self.placeholders[0]
        plh.qbits = q_bits_activation
        plh.scale, plh.zerop, plh.qmin, plh.qmax, plh.dtype = get_linear_quant_params_from_tensor(
            plh, q_mode_activation, plh.qbits, is_signed=out_signed)
        # out_scale = (1.0 if out.qinvariant else plh.scale)
        out_scale = out.scale * fused_multiplier
    if is_torch_tensor_with_multi_data(inp0.scale) or is_torch_tensor_with_multi_data(inp1.scale):
        dev = out.device
        inp0_scale = torch_tensor(inp0.scale, device=dev)
        inp1_scale = torch_tensor(inp1.scale, device=dev)

        if self.attrs.get('splitted_matmul', False):
            inp0_scale_base = inp0_scale.max() * 0.9
            inp1_scale_base = inp1_scale.max() * 0.9
            ts = 1. / inp0_scale_base / inp1_scale_base

            re_inp0_scale = inp0_scale_base / inp0_scale
            re_inp1_scale = inp1_scale_base / inp1_scale
            inp_scale_matrix = re_inp0_scale.reshape([-1, 1]) * re_inp1_scale.reshape([1, -1])
            ts_matrix = inp_scale_matrix

            # in set_tensor_quantization_attrs, the matmul has not child, so firstly check this condition.
            if len(self.children) > 0:
                mul_node = self.children[0]
                const_node = mul_node.parents[0] if mul_node.parents[0].type == OpType.Constant else mul_node.parents[1]

                const_node.constants['weights'].betensor = ts_matrix
                const_w_t = const_node.constants['weights']
                q_bits_weight = const_node.attrs['q_bits_weight']
                q_mode_weight = const_node.attrs['q_mode_weight']
                const_w_t.statistic(1.0, None, reset=True)
                const_w_t.qbits = q_bits_weight
                apply_calibration_strategy(const_w_t,
                                           const_node.attrs['q_strategy_weight'],
                                           q_mode_weight)
                const_out_t = const_node.outputs[0]
                const_out_t.min = const_w_t.min
                const_out_t.max = const_w_t.max
                const_out_t.min_key_axis = const_w_t.min_key_axis
                const_out_t.max_key_axis = const_w_t.max_key_axis
                # from AIPUBuilder.Optimizer.ops.constant import constant_quantize
                # constant_quantize(const_node)
                const_node.quantized = True

                res = get_linear_quant_params_from_tensor(const_w_t,
                                                          q_mode_weight,
                                                          q_bits_weight,
                                                          is_signed=False)
                const_w_t.scale, const_w_t.zerop, const_w_t.qmin, const_w_t.qmax, const_w_t.dtype = res
                const_out_t.scale, const_out_t.zerop, const_out_t.qmin, const_out_t.qmax, const_out_t.dtype = res
                const_w_t.betensor = linear_quantize_clip(const_w_t.betensor,
                                                          const_w_t.broadcast_scale,
                                                          const_w_t.broadcast_zerop,
                                                          const_w_t.qmin,
                                                          const_w_t.qmax)
                out.scale, out.zerop = 1.0, 0
        else:

            inp0_scale = inp0_scale.reshape([-1, 1])  # pylint: disable=no-member
            inp1_scale = inp1_scale.reshape([1, -1])  # pylint: disable=no-member
            inp_scale_matrix = torch.matmul(inp0_scale, inp1_scale)
            ts = out_scale / inp_scale_matrix

    else:
        ts = out_scale / (inp0.scale * inp1.scale)

    do_scale, do_scale_type, do_shift, do_shift_type = \
        get_scale_approximation_params(ts,
                                       multiplier_bits,
                                       force_shift_positive=self.force_shift_positive)
    doscale_name = 'scale' if is_torch_tensor_with_multi_data(do_scale) else 'scale_value'
    doshift_name = 'shift' if is_torch_tensor_with_multi_data(do_shift) else 'shift_value'
    self.set_ir_field(doscale_name, do_scale, do_scale_type)
    self.set_ir_field(doshift_name, do_shift, do_shift_type)
    if not is_torch_tensor_with_multi_data(do_scale):
        self.params["shift_type"] = do_shift_type
        self.params["scale_type"] = do_scale_type

    if 'remain_shift' in self.attrs:
        self.params['remain_shift'] = self.attrs['remain_shift']
