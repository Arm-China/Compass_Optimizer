# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import torch


@op_register(OpType.CacheUpdate)
def cacheupdate(self, *args):
    # if is aipuopt forward, do not use global mem feature
    use_mem_cache = True
    if "q_bits_activation" in self.attrs:
        use_mem_cache = False

    if not use_mem_cache:
        history = self.inputs[0].betensor.clone()
        origin_shape = history.shape
        position = self.inputs[1].betensor
        value = self.inputs[2].betensor

        # cache seq_len dim is 2, and written in OP Specification
        batch, head_num, cache_len, head_dim = value.shape
        history = history.reshape(batch, head_num, -1, head_dim)
        history[:, :, position[1]:position[1]+cache_len, :] = value

        self.outputs[0].betensor = history
        self.outputs[1].betensor = history.clone().reshape(origin_shape)
        return

    if 'global_mem_cache' not in self.attrs:
        self.attrs['global_mem_cache'] = {}
    if not self.inputs[0].name in self.attrs['global_mem_cache']:
        self.attrs['global_mem_cache'][self.inputs[0].name] = self.inputs[0].betensor.clone()
    global_mem = self.attrs['global_mem_cache'][self.inputs[0].name]

    blk_idx = self.params['block_index']
    buf = global_mem
    idx = self.inputs[1].betensor.long()[0]
    inp = self.inputs[2].betensor.type_as(buf)

    bsz, head_num, cache_len, head_dim = inp.shape
    queue_in = inp.permute(0, 2, 1, 3).reshape(1, 1, cache_len, head_num * head_dim)
    buf[blk_idx:blk_idx+1, 0:1, idx, :] = queue_in

    if len(idx) == 1:
        out_idx = torch.arange(idx.item()+1).long().to(inp.device)
    else:
        out_idx = idx
    out = buf[blk_idx:blk_idx+1, 0:1, out_idx, :].reshape(bsz, len(out_idx), head_num, head_dim)
    out = out.permute(0, 2, 1, 3)
    self.outputs[0].betensor = out
    self.outputs[1].betensor = global_mem
    return (self.outputs[0].betensor, self.outputs[1].betensor)


@quant_register(OpType.CacheUpdate)
def cacheupdate_quantize(self, *args):
    inp = self.inputs[2]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    out = self.outputs[1]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
