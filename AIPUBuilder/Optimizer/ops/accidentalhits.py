# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch


@op_register(OpType.AccidentalHits)
def accidentalhits(self, *args):
    # parser has handles this max(32768, len(batch_num*num_true))
    max_out_len = self.outputs[0].ir_shape[0]

    # only support integer of true_classes and sampled_candidate
    true_classes = self.inputs[0].betensor.to(torch.int64)
    sampled_candidate = self.inputs[1].betensor.to(torch.int64)
    batch_size = true_classes.shape[0]
    num_true = true_classes.shape[1]
    dev = true_classes.device
    out_len = 0
    out1 = torch.zeros(max_out_len, device=dev)
    out2 = torch.zeros(max_out_len, device=dev)

    outt1, outt2 = [], []
    yy = sampled_candidate.unsqueeze(0).repeat(true_classes.shape[1], 1)
    for d in range(true_classes.shape[0]):
        xx = true_classes[d].unsqueeze(-1).repeat(1, yy.shape[1])
        idx = torch.nonzero(torch.eq(xx, yy))
        if idx.numel() > 0:
            outt2.append(idx[:, 1])
            outt1.append(torch.ones_like(idx[:, 1]) * d)

    if outt1:
        outt1 = torch.hstack(outt1)
        outt2 = torch.hstack(outt2)
        out_len = min(len(outt1), max_out_len)
        out1[:out_len] = outt1[:out_len]
        out2[:out_len] = outt2[:out_len]

    self.outputs[0].betensor = out1
    self.outputs[1].betensor = out2
    self.outputs[2].betensor = torch.tensor([out_len])

    return [o.betensor for o in self.outputs]


@quant_register(OpType.AccidentalHits)
def accidentalhits_quantize(self, *args):
    """
    we resume the self.outputs[2].size < 65535
    :param self:
    :param args:
    :return:
    """
    for inp, out in zip(self.inputs, self.outputs[:2]):
        out.scale = 1.
        out.zerop = 0
        out.qinvariant = True

        out_numel = inp.ir_shape[0]
        out_bits, _ = range2dtype(0, out_numel, force_int=self.force_dtype_int)
        out.qbits = max(16, out_bits)
        out.dtype = bits2dtype(out.qbits, False or self.force_dtype_int)

    q_bits_activation = self.attrs["q_bits_activation"]
    out = self.outputs[2]
    out.scale = 1.
    out.zerop = 0
    out.qbits = max(16, q_bits_activation)
    out.dtype = bits2dtype(out.qbits, is_signed=False or self.force_dtype_int)
    out.qinvariant = True
