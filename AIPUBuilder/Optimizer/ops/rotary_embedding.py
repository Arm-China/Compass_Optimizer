# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import torch

register_optype('RotaryEmbedding')


@op_register(OpType.RotaryEmbedding)
def rotary_embedding_forward(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    x = inp.betensor
    batch_size = x.shape[0]
    sequence_length = x.shape[-2]
    support3d = False
    if len(x.shape) == 3:
        support3d = True
        hidden_size = x.shape[2]
        num_heads = self.params['num_heads']
        head_size = int(hidden_size / num_heads)
        new_shape = [batch_size, sequence_length, num_heads, head_size]
        x = x.reshape(new_shape)
        x = x.permute(0, 2, 1, 3)
    head_size = x.shape[3]
    rotary_embedding_dim = self.get_param('rotary_embedding_dim', optional=True, default_value=head_size)
    if rotary_embedding_dim < 1:
        # If rotary_embedding_dim not provided, perform full rotation by using head_size
        rotary_embedding_dim = head_size
    x_rotate = x[:, :, :, :rotary_embedding_dim]
    x_not_rotate = x[:, :, :, rotary_embedding_dim:]
    rotary_embedding_dim_half = int(rotary_embedding_dim / 2)
    # cos = self.constants["cos"]
    # sin = self.constants["sin"]
    cos = self.inputs[1]
    sin = self.inputs[2]
    if len(self.inputs) > 3:
        position_ids = self.inputs[3].betensor.long()
    else:
        position_ids = torch.arange(sequence_length, device=x.device).repeat(batch_size, 1)
    if 2 == len(cos.betensor.shape):
        # 2D cos cache
        cos_cache = cos.betensor.unsqueeze(0).repeat(batch_size, 1, 1)
        sin_cache = sin.betensor.unsqueeze(0).repeat(batch_size, 1, 1)
    else:
        # 3D cos cache
        cos_cache = cos.betensor
        sin_cache = sin.betensor
    cos_v = cos_cache[torch.arange(batch_size)[:, None], position_ids]
    sin_v = sin_cache[torch.arange(batch_size)[:, None], position_ids]
    cos_v = cos_v[:, :, :rotary_embedding_dim_half].unsqueeze(1)
    sin_v = sin_v[:, :, :rotary_embedding_dim_half].unsqueeze(1)
    interleaved = self.get_param('interleaved', optional=True, default_value=0)
    if interleaved:
        x1 = x_rotate[:, :, :, 0::2]
        x2 = x_rotate[:, :, :, 1::2]
    else:
        x1, x2 = torch.split(x_rotate, rotary_embedding_dim_half, dim=-1)
    # Calculate real and imaginary values
    if self.quantized:
        # requant_scale = self.get_ir_field(['scale', 'scale_value'], default_value=1)
        # requant_shift = self.get_ir_field(['shift', 'shift_value'], default_value=0)
        # real = linear_requantize(real, requant_scale, requant_shift, out.zerop, out.qmin,
        #                          out.qmax, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', out.dtype))
        # imag = linear_requantize(imag, requant_scale, requant_shift, out.zerop, out.qmin,
        #                          out.qmax, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', out.dtype))
        # requant_shift = self.params["not_rotate_shift_value"]
        # requant_scale = self.params["not_rotate_scale_value"]
        # x_not_rotate = linear_requantize(x_not_rotate, requant_scale, requant_shift, out.zerop, out.qmin,
        #                                  out.qmax, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', out.dtype))
        rotate_rescale_cos = self.params['rotate_rescale_cos']
        rotate_rescale_sin = self.params['rotate_rescale_sin']
        not_rotate_rescale = self.params['not_rotate_rescale']
        real = cos_v.long() * x1.long() * rotate_rescale_cos - sin_v.long() * x2.long() * rotate_rescale_sin
        imag = sin_v.long() * x1.long() * rotate_rescale_sin + cos_v.long() * x2.long() * rotate_rescale_cos
        real = linear_quantize_clip(real.float(
        ), 1.0, 0, self.outputs[0].qmin, self.outputs[0].qmax, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', self.outputs[0].dtype)).long()
        imag = linear_quantize_clip(imag.float(
        ), 1.0, 0, self.outputs[0].qmin, self.outputs[0].qmax, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', self.outputs[0].dtype)).long()
        x_not_rotate = linear_quantize_clip(x_not_rotate.float(
        ), not_rotate_rescale, 0, self.outputs[0].qmin, self.outputs[0].qmax, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', self.outputs[0].dtype)).long()
    else:
        real = cos_v.float() * x1.float() - sin_v.float() * x2.float()
        imag = sin_v.float() * x1.float() + cos_v.float() * x2.float()

    # Inserted rotated embeddings back to the original input
    if interleaved:
        x_rotate_concat = torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1)
        x_rotate = x_rotate_concat.reshape(x_rotate.shape)
    else:
        x_rotate = torch.cat([real, imag], dim=-1)
    y = torch.cat([x_rotate, x_not_rotate], dim=-1)
    if support3d:
        y = y.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
    out.betensor = y
    return out.betensor


@quant_register(OpType.RotaryEmbedding)
def rotary_embedding_quantize(self, *args):
    # will not have a quantized RoPE, just for opt flow
    q_mode_activation = self.attrs["q_mode_activation"]
    q_mode_weight = self.attrs["q_mode_weight"]
    q_bits_weight = self.attrs["q_bits_weight"]
    q_bits_activation = self.attrs["q_bits_activation"]
    multiplier_bits = self.get_attrs('multiplier_bits', optional=True, default_value=q_bits_activation)

    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    # cos = self.constants["cos"]
    # sin = self.constants["sin"]
    # w = PyTensor("tmp")
    # w.max = cos.max if cos.max > sin.max else sin.max
    # w.min = cos.min if cos.min < sin.min else sin.min
    # if cos.max_key_axis is not None:
    #     cos.max_key_axis = torch.where(cos.max_key_axis > sin.max_key_axis, cos.max_key_axis, sin.max_key_axis)
    #     cos.min_key_axis = torch.where(cos.min_key_axis < sin.min_key_axis, cos.min_key_axis, sin.min_key_axis)
    inp = self.inputs[0]
    out = self.outputs[0]
    out.qbits = q_bits_activation
    out.qinvariant = False
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, q_bits_activation, is_signed=True)
    # cos.scale, cos.zerop, cos.qmin, cos.qmax, cos.dtype = get_linear_quant_params_from_tensor(w, q_mode_weight, q_bits_weight,
    #                                                                                           is_signed=True)
    # sin.scale, sin.zerop, sin.qmin, sin.qmax, sin.dtype = cos.scale, cos.zerop, cos.qmin, cos.qmax, cos.dtype
    # sin.qbits = q_bits_weight
    # cos.qbits = q_bits_weight
    # sin.qinvariant = False
    # cos.qinvariant = False
    # sin.betensor = linear_quantize_clip(sin.betensor, sin.broadcast_scale, sin.broadcast_zerop, sin.qmin, sin.qmax)
    # cos.betensor = linear_quantize_clip(cos.betensor, cos.broadcast_scale, cos.broadcast_zerop, cos.qmin, cos.qmax)
    # local_rescale = out.scale / (inp.scale * cos.scale)
    # do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(local_rescale,
    #                                                                                   mult_bits=multiplier_bits,
    #                                                                                   force_shift_positive=self.force_shift_positive)
    # doscale_name = 'scale' if is_torch_tensor_with_multi_data(do_scale) else 'scale_value'
    # doshift_name = 'shift' if is_torch_tensor_with_multi_data(do_shift) else 'shift_value'
    # if not is_torch_tensor_with_multi_data(do_scale):
    #     self.params["shift_type"] = do_shift_type
    #     self.params["scale_type"] = do_scale_type
    # else:
    #     do_scale = do_scale.reshape(1, 1, do_scale.numel(), 1)
    #     do_shift = do_shift.reshape(1, 1, do_shift.numel(), 1)
    # self.set_ir_field(doscale_name, do_scale, do_scale_type)
    # self.set_ir_field(doshift_name, do_shift, do_shift_type)

    # not_rotate_rescale = out.scale / inp.scale
    # do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(not_rotate_rescale,
    #                                                                                   mult_bits=multiplier_bits,
    #                                                                                   force_shift_positive=self.force_shift_positive)
    # self.params["not_rotate_shift_type"] = do_shift_type
    # self.params["not_rotate_scale_type"] = do_scale_type
    # self.params["not_rotate_shift_value"] = do_shift
    # self.params["not_rotate_scale_value"] = do_scale

    cos = self.inputs[1]
    sin = self.inputs[2]
    self.params['rotate_rescale_cos'] = out.scale / (inp.scale * cos.scale)
    self.params['rotate_rescale_sin'] = out.scale / (inp.scale * sin.scale)
    self.params['not_rotate_rescale'] = out.scale / inp.scale
