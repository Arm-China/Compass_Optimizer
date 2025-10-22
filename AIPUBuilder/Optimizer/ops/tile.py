# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_WARN


@op_register(OpType.Tile)
def tile(self, *args):
    inp = self.inputs[0].betensor
    out = self.outputs[0]
    # _reps = [oshape // ishape for oshape, ishape in zip(out.shape, inp.shape)]
    reps = self.get_param('repeats')
    if inp.dim() != len(reps):
        OPT_WARN('please check the dim between input.dim and len(repeats) in Tile Op')
    out.betensor = inp.repeat(reps)
    return out.betensor


@quant_register(OpType.Tile)
def tile_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.qinvariant = inp.qinvariant
    out.qmin = inp.qmin
    out.qmax = inp.qmax

    if out.key_axis is not None:
        ka = out.key_axis
        if inp.key_axis is not None:
            rep = self.params['repeats'][ka]
        else:
            rep = out.ir_shape[ka]
        out.scale = inp.scale.repeat(rep)
        out.zerop = inp.zerop.repeat(rep)
