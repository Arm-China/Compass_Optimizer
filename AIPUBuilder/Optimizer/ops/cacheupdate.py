# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import torch

shared_mem = {}


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

    if not self.graph in shared_mem:
        shared_mem[self.graph] = []
    if not "mem_idx" in self.attrs:
        self.attrs["mem_idx"] = len(shared_mem[self.graph])
        shared_mem[self.graph].append(torch.zeros(self.inputs[0].ir_shape))
    if self.inputs[0].betensor.numel() > 1:
        shared_mem[self.graph][self.attrs["mem_idx"]] = self.inputs[0].betensor
    batch, num_head, cache_len, head_dim = self.inputs[2].betensor.shape
    pos_idx = self.inputs[1].betensor[0, 0].item()
    shared_mem[self.graph][self.attrs["mem_idx"]].view(
        [batch, num_head, -1, head_dim])[:, :, pos_idx:pos_idx+cache_len, :] = self.inputs[2].betensor
    self.outputs[0].betensor = shared_mem[self.graph][self.attrs["mem_idx"]].reshape(
        [batch, num_head, -1, head_dim])[:, :, :pos_idx+cache_len]
    self.outputs[1].betensor = shared_mem[self.graph][self.attrs["mem_idx"]]


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
