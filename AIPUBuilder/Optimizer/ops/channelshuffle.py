# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import OPT_ERROR, OPT_WARN
import torch


@op_register(OpType.ChannelShuffle)
def channelshuffle(self, *args):
    """
    now only support [n, h, w, c] input shape
    torch.nn.functional.channel_shuffle is alpha version in torch 1.7.1
    :param self:
    :param args:
    :return:
    """
    inp_t = self.inputs[0].betensor
    group = self.get_param('group')
    splits = self.get_param('splits')

    inp_shape = list(inp_t.shape)
    trans_perm = list(range(len(inp_shape) + 1))
    trans_perm[-1], trans_perm[-2] = trans_perm[-2], trans_perm[-1]
    new_shape = inp_shape[:-1]
    new_shape += [group, inp_shape[-1] // group]
    out = inp_t.reshape(new_shape)
    out = torch.permute(out, trans_perm)
    out = out.reshape(inp_shape)

    if splits != 1:
        split_size = out.shape[-1] // splits
        out = torch.split(out, split_size, dim=-1)
        for i, o in enumerate(out):
            self.outputs[i].betensor = o
    else:
        self.outputs[0].betensor = out

    return out


@quant_register(OpType.ChannelShuffle)
def channelshuffle_quantize(self, *args):
    inp = self.inputs[0]
    for out in self.outputs:
        out.dtype = inp.dtype
        out.scale = inp.scale
        out.zerop = inp.zerop
        out.qbits = inp.qbits
        out.qmin = inp.qmin
        out.qmax = inp.qmax
        out.qinvariant = inp.qinvariant
