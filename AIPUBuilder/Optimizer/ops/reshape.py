# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch


@op_register(OpType.Reshape)
def reshape(self, *args):
    try:
        inp = self.inputs[0].betensor.clone()
        out = self.outputs[0]
        shape = list(out.ir_shape)[:]

        ir_batch = self.attrs['batch_size_in_IR'] if 'batch_size_in_IR' in self.attrs else 1
        batch_axis = 0
        for idx, ss in enumerate(shape):
            if ss == ir_batch:
                shape[idx] = -1
                batch_axis = idx
                break
        # shape[0] = -1 # left batch -1
        if batch_axis != 0:
            OPT_WARN('layer_id=%s, type=%s, the batch axis is not 0, making the result unpredictable possibly' %
                     (self.attrs['layer_id'], str(self.type)), log_once=True)
        if ir_batch != 0 and len(shape) != 0:
            # have batch_dim
            shape[batch_axis] = -1
        out.betensor = torch.reshape(inp, shape)
        return out.betensor
    except Exception as e:
        inp = self.inputs[0].betensor.clone()
        real_inshape = list(inp.shape)
        shape = list(out.ir_shape)
        ir_inshape = list(self.inputs[0].ir_shape)
        ir_outshape = shape[:]
        diff_dim = -1
        for i in range(len(real_inshape)):
            if ir_inshape[i] != real_inshape[i]:
                diff_dim = i
                break

        if diff_dim >= 0:
            in_idx = 0
            in_size = 1
            out_idx = 0
            out_size = 1
            axis = -1
            while in_idx < len(ir_inshape):
                in_size *= ir_inshape[in_idx]
                while out_idx < len(ir_outshape):
                    if out_size >= in_size:
                        break
                    else:
                        out_size *= ir_outshape[out_idx]
                    if out_size == in_size and in_idx == diff_dim:
                        axis = out_idx
                        break
                    out_idx += 1
                in_idx += 1
                if axis >= 0:
                    break
            shape[axis] = real_inshape[diff_dim]
        out.betensor = torch.reshape(inp, shape)
        return out.betensor


@quant_register(OpType.Reshape)
def reshape_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    out.qmin = inp.qmin
    out.qmax = inp.qmax
    if inp.key_axis is not None and inp.key_axis_g > 1:
        out.scale = out.scale[::inp.key_axis_g]
        out.zerop = out.zerop[::inp.key_axis_g]

    if out.key_axis is not None and out.key_axis_g > 1:
        out.scale = torch.tile(out.scale, (out.key_axis_g,))
        out.zerop = torch.tile(out.zerop, (out.key_axis_g,))
