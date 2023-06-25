# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch


@op_register(OpType.ZeroFraction)
def zerofraction(self, *args):
    batch_size_in_IR = self.get_attrs("batch_size_in_IR", optional=True, default_value=1)
    inp_t = self.inputs[0].betensor + self.inputs[0].zerop

    if batch_size_in_IR == 0:
        nonzero_num = torch.count_nonzero(inp_t).item()
        batch_total_num = inp_t.numel()
        total_zero_num = torch.tensor(batch_total_num - nonzero_num, device=inp_t.device)
    else:
        batch = self.inputs[0].betensor.shape[0]
        batch_total_num = inp_t.numel() // batch
        total_zero_num = torch.zeros([batch], device=inp_t.device)
        for b in range(batch):
            batch_data = inp_t[b]
            nonzero_num = torch.count_nonzero(batch_data).item()
            total_zero_num[b] = batch_total_num - nonzero_num

    if self.quantized:
        enlarge_scale = self.get_param('output_scale')
        zero_fraction = torch.div((total_zero_num * enlarge_scale + batch_total_num//2),
                                  batch_total_num, rounding_mode='floor').int()
        zero_fraction -= self.outputs[0].zerop
    else:
        zero_fraction = total_zero_num / batch_total_num

    self.outputs[0].betensor = zero_fraction
    return self.outputs[0].betensor


@quant_register(OpType.ZeroFraction)
def zerofraction_quantize(self, *args):
    q_bits_activation = self.attrs["q_bits_activation"]
    enlarge_scale = 2 ** q_bits_activation - 1
    self.params['output_scale'] = enlarge_scale

    out = self.outputs[0]
    out.qinvariant = False
    out.qbits = q_bits_activation
    out.dtype = bits2dtype(q_bits_activation, False or self.force_dtype_int)
    out.qmin, out.qmax = dtype2range(out.dtype)
    out.scale = enlarge_scale
    out.zerop = 0
    if self.force_dtype_int:
        out.zerop = - out.qmin
