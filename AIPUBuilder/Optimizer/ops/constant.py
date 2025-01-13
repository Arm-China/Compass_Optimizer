# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils.quant_tool_utils import *
from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.ops.inp import inp_quantize


@op_register(OpType.Constant)
def constant(self, *args):
    t = self.constants["weights"].betensor.clone()
    batch_size_in_IR = self.get_attrs("batch_size_in_IR", optional=True, default_value=1)
    if batch_size_in_IR != 0 and self.current_batch_size > 1 and t.dim() > 1 and t.shape[
            0] == batch_size_in_IR and 0 == int(self.current_batch_size) % int(t.shape[0]):
        # multi-batch: tile constant values
        tshape = list(t.shape)
        wshape = list(t.shape)
        tshape[0] = self.current_batch_size
        reps = [oshape // ishape for oshape, ishape in zip(tshape, t.shape)]
        t = t.repeat(reps)
        OPT_DEBUG(f"layer_id={self.attrs['layer_id']}, {str(self.type)}, {self.name} : "
                  f"repeat its contents from shape {wshape} to {tshape}", log_once=True)
    self.outputs[0].betensor = t
    return self.outputs[0].betensor


@quant_register(OpType.Constant)
def constant_quantize(self, *args):
    inp_quantize(self, *args)
    w = self.constants["weights"]
    out = self.outputs[0]
    if not torch.equal(w.betensor, w.betensor.new_zeros(w.betensor.shape)):
        out.set_qinvariant()
    w.scale = out.scale
    w.zerop = out.zerop
    w.qbits = out.qbits
    w.dtype = out.dtype
    w.qmin = out.qmin
    w.qmax = out.qmax
    w.qinvariant = out.qinvariant
    w.betensor = linear_quantize_clip(w.betensor, out.broadcast_scale, out.broadcast_zerop, w.qmin, w.qmax)
