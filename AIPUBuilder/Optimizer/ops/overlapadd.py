# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.utils.quant_tool_utils import *
from AIPUBuilder.Optimizer.utils.string_utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import OPT_ERROR


@op_register(OpType.OverlapAdd)
def overlapadd(self, *args):
    inp = self.inputs[0].betensor
    if self.quantized:
        inp += torch.tensor(self.inputs[0].zerop, device=inp.device).int()
    if len(inp.shape) < 2:
        OPT_ERROR("input tensor must be at least rank 2")
    frame_step = self.get_param('frame_step') if 'frame_step' in self.params else 1
    shape_list = list(inp.size())
    frame_length = shape_list[-1]
    if frame_step > frame_length:
        OPT_ERROR("frame step must be less than or equal to frame length")
    frames = shape_list[-2]
    output_size = (frames-1)*frame_step+frame_length
    output_shape = [shape_list[i] for i in range(len(shape_list)-2)]
    output_shape.append(output_size)
    outp = torch.zeros(output_shape, device=inp.device)
    for j in range(frame_length):
        for index in range(frames):
            for i in range(output_size):
                if j+frame_step*index == i:
                    outp[..., i] += inp[..., index, j]

    if self.quantized:
        shift = self.params["shift_value"]
        scale = self.params["scale_value"]
        outp = linear_requantize(outp, scale, shift, self.outputs[0].zerop, self.outputs[0].qmin, self.outputs[0].qmax)
    self.outputs[0].betensor = outp
    return outp


@quant_register(OpType.OverlapAdd)
def overlapadd_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    q_bits_activation = self.attrs["q_bits_activation"]
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    if inp.qinvariant:
        out.scale = 1
        out.zerop = 0
        out.qbits, out.dtype = range2dtype(out.extrema_min, out.extreama_max, force_int=self.force_dtype_int)
        out.qinvariant = True
        out.qmin, out.qmax = dtype2range(out.dtype)
        do_scale, do_shift = 1, 0
    else:
        out_signed = is_signed(inp.dtype)
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, q_bits_activation, is_signed=out_signed)
        out.qbits = q_bits_activation
        out.qinvariant = inp.qinvariant
        local_scale = out.scale / inp.scale
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(local_scale, mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)
    self.params["shift_value"] = int(do_shift)
    self.params["shift_type"] = do_shift_type
    self.params["scale_value"] = int(do_scale)
    self.params["scale_type"] = do_scale_type
