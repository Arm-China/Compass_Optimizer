# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
import torch

register_optype('Dilation')


def dilation_erosion_fun(self, padding_value, compare_func, weight_reverse=False):
    out = self.outputs[0]
    inp = self.inputs[0]
    input_data = inp.betensor.float()
    weights = self.constants['weights'].betensor.clone().float().permute([3, 1, 2, 0])  # [C,H,W,1]=>[1,H,W,C]
    if weight_reverse:
        weights = -torch.flip(weights, [1, 2])

    batch = input_data.shape[0]
    ih = self.inputs[0].ir_shape[1]
    iw = self.inputs[0].ir_shape[2]
    oh = self.outputs[0].ir_shape[1]
    ow = self.outputs[0].ir_shape[2]
    oc = self.outputs[0].ir_shape[3]

    kernel_size = self.get_param('kernel_y', optional=True, default_value=1), self.get_param("kernel_x")
    stride = self.get_param("stride_y", optional=True, default_value=1), self.get_param("stride_x")
    dilation = self.get_param("dilation_y", optional=True, default_value=1), self.get_param("dilation_x")
    pad_left = self.get_param('pad_left')
    pad_right = self.get_param('pad_right')
    pad_top = self.get_param('pad_top')
    pad_bottom = self.get_param('pad_bottom')

    padding = (0, 0,  # c
               pad_left, pad_right,  # w
               pad_top, pad_bottom  # h
               )

    inp_zerop = self.inputs[0].zerop if self.quantized else 0
    input_data += inp_zerop
    input_data = torch.nn.functional.pad(input_data, padding, value=padding_value)

    outp = torch.zeros([batch, oh, ow, oc], device=input_data.device)
    weights = torch.repeat_interleave(weights, batch, dim=0)
    startX = torch.arange(ow) * stride[1]
    startY = torch.arange(oh) * stride[0]
    endY = startY + (kernel_size[0]-1)*dilation[0]+1
    endX = startX + (kernel_size[1]-1)*dilation[1]+1

    if self.quantized:
        scale, scale0, scale1 = self.params["scale_value"]
        shift = self.params["shift_value"]
        input_data = input_data * scale0
        weights = weights * scale1
        qmin, qmax = self.outputs[0].qmin, self.outputs[0].qmax
        out_zerop = self.outputs[0].zerop

    for cy in range(oh):
        for cx in range(ow):
            tmp_data = input_data[:, startY[cy]:endY[cy]:dilation[0], startX[cx]:endX[cx]:dilation[1], :]
            tmp_data = tmp_data + weights
            max_value = compare_func(tmp_data, (1, 2))
            outp[:, cy, cx, :] = max_value

    if self.quantized:
        outp = linear_requantize(outp, scale, shift, out_zerop, qmin, qmax)

    return outp


@op_register(OpType.Dilation)
def dilation(self, *args):
    outp = dilation_erosion_fun(self, padding_value=float('-inf'), compare_func=torch.amax, weight_reverse=False)
    self.outputs[0].betensor = outp
    return outp


@quant_register(OpType.Dilation)
def dilation_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    q_mode_weight = self.attrs["q_mode_weight"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    out = self.outputs[0]
    w = self.constants['weights']

    w_signed = True if w.extrema_min < 0 else False
    w.scale, w.zerop, w.qmin, w.qmax, w.dtype = get_linear_quant_params_from_tensor(
        w, QuantMode.to_per_tensor(q_mode_weight), q_bits_activation, is_signed=True)
    w.betensor = linear_quantize_clip(w.betensor, w.scale, w.zerop, w.qmin, w.qmax)
    w.qbits = q_bits_activation
    w.qinvariant = False

    out_signed = is_signed(inp.dtype) or w_signed or self.type == OpType.Erosion
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, q_bits_activation, is_signed=out_signed)
    out.qbits = q_bits_activation
    out.qinvariant = False

    g_eltwise_scale_bits = 8
    inp_scale_max = max(inp.scale, w.scale)
    # due to aiff don't support uint16 max 65535,so we use INT16 replace UINT16
    _, clip_max = dtype2range(Dtype.INT16)
    # avoid to warning occurrence, we need ignore the relative extreme big/small scale, so
    # proof_min_ration is defined
    proof_min_ration = (2**g_eltwise_scale_bits)/clip_max  # 2**-7
    inp0_scale = inp.scale
    inp1_scale = w.scale
    if inp.scale/w.scale < proof_min_ration or inp.scale/w.scale < proof_min_ration:
        inp0_scale = min(max(inp.scale, 1./clip_max), clip_max)
        inp1_scale = min(max(w.scale, 1./clip_max), clip_max)
        inp_scale_max = max(inp0_scale, inp1_scale)

    local_rescale = out.scale / (inp_scale_max)
    scale0 = (inp_scale_max / inp0_scale) * (2**g_eltwise_scale_bits)
    scale1 = (inp_scale_max / inp1_scale) * (2**g_eltwise_scale_bits)

    while int(scale0) > clip_max or int(scale1) > clip_max:
        if scale0 == 1:  # scale1>clip_max, but scale0=1,
            (OPT_DEBUG('layer_id=%s, layer_type=%s the weight scale=%d has out range [0, %d], please attention.'
                       % (self.attrs['layer_id'], str(self.type), int(scale1), int(clip_max))))
        if scale1 == 1:
            (OPT_DEBUG('layer_id=%s, layer_type=%s the input scale=%d has out range [0, %d], please attention.'
                       % (self.attrs['layer_id'], str(self.type), int(scale0), int(clip_max))))
        scale0 = max(round(scale0 / 2), 1)  # to avoid one scale to be 0
        scale1 = max(round(scale1 / 2), 1)  # to avoid one scale to be 0
        g_eltwise_scale_bits -= 1
    # pass the scale factor for ideal mode case
    do_scale, do_scale_type, do_shift, do_shift_type = \
        get_scale_approximation_params(local_rescale / (2**g_eltwise_scale_bits),
                                       mult_bits=q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)
    self.params["shift_value"] = int(do_shift)
    self.params["shift_type"] = do_shift_type
    self.params["scale_value"] = [int(do_scale), int(scale0), int(scale1)]
    self.params["scale_type"] = [do_scale_type, Dtype.UINT16, Dtype.UINT16]
