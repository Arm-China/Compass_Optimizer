# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch

###########################################################
# layer_id=2
# layer_name=LeftShift
# layer_type=BitShift
# layer_bottom=[Placeholder_0,Placeholder_1_0]
# layer_bottom_shape=[[2,3,4],[2,3,4]]
# layer_bottom_type=[int8,int8]
# layer_top=[LeftShift_0]
# layer_top_shape=[[2,3,4]]
# layer_top_type=[int8]
# direction=LEFT

#input = 0b111(int8)
# input >> 2 = 0b1 = 1
# input << 5 = 0b11100000 = -32
###########################################################


@quant_register(OpType.BitShift)
def bitshift_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]

    str_type = self.attrs['layer_top_type_original'][0]
    out_dtype = str2dtype(str_type)
    out_sign = is_signed(out_dtype)
    if not inp0.qinvariant or not inp1.qinvariant:
        OPT_FATAL("the inputs of op{%s} don't be quantized." % (self.type))
    else:  # inp0.qinvariant is true and inp1.qinvariant is true
        qbits = dtype2bits(out_dtype)
        out.qbits = qbits
        out.scale = 1
        out.zerop = 0
        qmin, qmax = dtype2range(out_dtype)
        out.qmin = qmin
        out.qmax = qmax
        out.dtype = out_dtype
        out.qinvariant = True


@op_register(OpType.BitShift)
def bitshift(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]
    direction = self.get_param('direction').upper()

    input_data = inp0.betensor.clone()
    input_shift = inp1.betensor.long()
    # check valid
    if True in (input_shift < 0):
        OPT_WARN('%s{%s} Shift cannot contain negative numbers because hardware does not support negative numbers'
                 % (self.name, str(self.type)))

    if 'layer_top_type_original' in self.attrs:
        out_dtype = str2dtype(self.attrs['layer_top_type_original'][0])
    else:
        out_dtype = inp0.dtype

    out_sign = is_signed(out_dtype)
    _, qmax = bits2range(dtype2bits(out_dtype), False)
    np_out_dtype = dtype2nptype(out_dtype)
    input_data = input_data.cpu().numpy().astype(np_out_dtype)
    input_shift = input_shift.cpu().numpy()

    if direction == "LEFT":
        outp = input_data << input_shift
        outp = outp & qmax
    elif direction == "RIGHT":
        outp = input_data >> input_shift
    else:
        OPT_FATAL("unsupported method: %s for BitShift in node:%s" % (direction, self.name))

    outp = outp.astype(np_out_dtype)
    if not out_sign:
        pytensor = PyTensor('out', outp)
        outp = pytensor.betensor.to(inp0.betensor.device)
    else:
        outp = torch.tensor(outp, device=inp0.betensor.device).type(dtype2torch_type(out_dtype))
    self.outputs[0].betensor = outp
    return self.outputs[0].betensor
