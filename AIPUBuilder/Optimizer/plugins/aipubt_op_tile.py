# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *

# Delete the comment on the next line to enable this plugin
# @op_register(OpType.Tile)


def tile(self, *args):
    if 'multipliers' in self.params:
        key = 'multiplier'
    elif 'reps' in self.params:
        key = 'reps'
    else:
        OPT_ERROR("Tile op needs 'multipliers' or 'reps' param.")

    reps = self.params[key]
    if isinstance(reps, str):
        reps = [int(r) for r in reps.split(',')]
    inp_t = self.inputs[0].betensor
    out_t = inp_t.repeat(reps)
    self.outputs[0].betensor = out_t

    return out_t

# Delete the comment on the next line to enable this plugin
# @op_register(OpType.Tile)


def tile_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
