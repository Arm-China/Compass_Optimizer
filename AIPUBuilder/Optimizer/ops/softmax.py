# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import *


@quant_register(OpType.Softmax)
def softmax_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Softmax currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]
    inp = self.inputs[0]
    dev = inp.betensor.device
    # so we allow to accumulate less than 2**12 items with a 32bit accumulator
    axis = self.get_param('axis')
    shape_value_in_axis = inp.ir_shape[axis]
    ibits = torch.ceil(torch.log2(torch.tensor(shape_value_in_axis))).long().item()
    scaling_bits = self.attrs['scaling_bits']
    mbits = max(8, 32-ibits) if scaling_bits[0] < 1 else scaling_bits[0]
    max_val = torch.tensor((1 << mbits)-1, device=dev)
    max_inp = torch.log(max_val)
    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    quant_range_linspace = torch.linspace(0, 2 ** inp.qbits - 1, steps=lsteps, device=dev)
    max_inp -= inp.zerop[0] / inp.scale[0]
    lut = linear_dequantize(quant_range_linspace - (2 ** inp.qbits - 1), inp.scale, inp.zerop) + max_inp
    lut = torch.exp(lut).round().clamp(0, max_val)
    self.constants["lut"] = PyTensor(
        self.name + "/explut", lut.cpu().numpy().astype(dtype2nptype(range2dtype(0, max_val.item())[1])))
    out = self.outputs[0]
    out_sign = False or self.force_dtype_int
    out.qbits = q_bits_activation
    out.qinvariant = False
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(self.outputs[0],
                                                                                              q_mode_activation,
                                                                                              out.qbits,
                                                                                              out_sign)
    do_scale_bits = out.qbits
    ss = get_scale_approximation_params(out.scale, do_scale_bits, force_shift_positive=self.force_shift_positive)
    do_scale, do_scale_type, do_shift, do_shift_type = ss
    self.params["shift_value"] = int(do_shift)
    self.params["shift_type"] = do_shift_type
    self.params["scale_value"] = int(do_scale)
    self.params["scale_type"] = do_scale_type
    self.params["quantize_method"] = 'LUT'


@op_register(OpType.Softmax)
def softmax(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    axis = self.get_param('axis')
    shape_value_in_axis = inp.betensor.shape[axis]
    in_size = inp.betensor.numel() / shape_value_in_axis
    if self.quantized:
        x = inp.betensor
        lut = self.constants["lut"].betensor
        max_v, _ = x.max(axis, keepdim=True)
        x = x - max_v + 2 ** inp.qbits - 1
        x = torch.reshape(x, (-1,))
        y = lookup_lut_powerof2(x,
                                lut,
                                inp.qbits,
                                False,
                                dtype2bits(self.constants["lut"].dtype),
                                is_signed(self.constants["lut"].dtype))
        y = torch.reshape(y, inp.betensor.shape)

        do_shift = self.params["shift_value"]
        do_scale = self.params["scale_value"]
        y_sum = y.sum(axis, keepdim=True, dtype=torch.long)

        '''
        now we support two kind of softmax quantized forward, the one shape_value_in_axis < 8 and in_size % 128 == 0 case
        existed is because of in small data numbers the saved enlarge_bits acurracy in division is faster than 31bits, the lib
        has better preformance, so in order to compatiblity the lib impl, we also has two kind of quantized forward.
        '''
        # 31->30: Adaptation the lut.max=1 case in android and these case lib would overflow using base_bits=31
        base_bits = 30 if lut.max() == 1 else 31
        if shape_value_in_axis < 8 and in_size % 128 == 0:
            scale_bits = torch.ceil(torch.log2(lut.max())).long()
            enlarge_bits = base_bits - scale_bits
            numerator = (y.long() << enlarge_bits) + (y_sum >> 1)
            denominator = torch.maximum(y_sum, torch.ones_like(y_sum))
            z = torch.div(numerator, denominator, rounding_mode='trunc')
            do_shift = do_shift + enlarge_bits
            out.betensor = linear_requantize(z, do_scale, do_shift, out.zerop, out.qmin, out.qmax)
        else:
            denominator = torch.maximum(y_sum, torch.ones_like(y_sum))
            scale_bits = torch.ceil(torch.log2(torch.tensor(do_scale))).long()
            enlarge_bits = base_bits - scale_bits
            enlarge_shift = do_shift + enlarge_bits
            enlarge_scale = do_scale * (2 ** enlarge_bits)
            reverse_sum = torch.div(enlarge_scale, denominator, rounding_mode='trunc')
            y_div_sum = y.long() * reverse_sum
            out.betensor = linear_requantize(y_div_sum, 1.0, enlarge_shift, out.zerop, out.qmin, out.qmax)
    else:
        out.betensor = torch.nn.functional.softmax(inp.betensor, dim=axis)
    return out.betensor
