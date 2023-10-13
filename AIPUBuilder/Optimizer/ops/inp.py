# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *


@op_register(OpType.Input)
def input_node(self, *args):
    if self.quantized:
        for o in self.outputs:
            if not (o.scale == 1 and o.zerop == 0):
                o.betensor = linear_quantize_clip(
                    o.betensor, o.scale, o.zerop, o.qmin, o.qmax)
    # Batch dimension does not exist in some speech models
    # The Batch dimension was expanded in dataset before and deleted here
    batch_size_in_IR = self.get_attrs("batch_size_in_IR", optional=True, default_value=1)
    if batch_size_in_IR == 0 and batch_size_in_IR != self.current_batch_size:
        for o in self.outputs:
            if len(o.ir_shape) != o.betensor.dim():
                o.betensor = o.betensor[0]
    return [o.betensor for o in self.outputs]


@quant_register(OpType.Input)
def inp_quantize(self, *args):
    out = self.outputs[0]
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Input currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]
    out_signed = True
    if out.extrema_min >= 0.0 and not self.force_dtype_int:
        out_signed = False
    out.qbits = q_bits_activation
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, is_signed=out_signed)
    out.qinvariant = False
    top_type_original = self.attrs['layer_top_type_original'][0]
    original_top_dtype = str2dtype(top_type_original)
    is_original_top_float = is_float(original_top_dtype)
    if False == is_original_top_float:
        OPT_DEBUG('This op do not need quantization, becasuse its orignal top type is %s' %
                  top_type_original, workflow_name='quantize', op_name=str(self.type))

        if self.type == OpType.Constant:
            out.qbits, out.dtype = range2dtype(out.extrema_min, out.extrema_max, force_int=self.force_dtype_int)
            out_signed = is_signed(out.dtype) or self.force_dtype_int
            out.qbits = max(out.qbits, q_bits_activation)
            out.dtype = bits2dtype(out.qbits, is_signed=out_signed)
        else:
            out.dtype = original_top_dtype
            if dtype2bits(original_top_dtype) > 32:
                out.dtype = bits2dtype(32, is_signed(original_top_dtype))
                OPT_WARN(
                    f"layer_id={self.attrs['layer_id']}, {str(self.type)}, {self.name} : its output dtype {original_top_dtype} is larger than 32bits, and will be clamped into 32bit", log_once=True)
            out.qbits = dtype2bits(out.dtype)
        out.qmin, out.qmax = dtype2range(out.dtype)
        out.scale = 1.0
        out.zerop = 0
        out.qinvariant = True
