# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import OPT_WARN, OPT_ERROR
import torch
from functools import reduce


def get_ds_shape(in_ir_shape, out_ir_shape, in_ds_shape):
    if len(in_ir_shape) != len(in_ds_shape):
        OPT_WARN(f"In Reshape Op: in_ir_shape={in_ir_shape} and in_ds_shape = {in_ds_shape}, which"
                 f"the length of runtime input shape(={len(in_ds_shape)}) != "
                 f"the length of input ir_shape(={len(in_ir_shape)})")

        in_ir_numel = reduce(lambda a, b: a*b, in_ir_shape)
        in_ds_numel = reduce(lambda a, b: a*b, in_ds_shape)
        # try to pop or insert 1 when ir_numel == ds_numel
        if in_ir_numel == in_ds_numel:
            if len(in_ir_shape) > len(in_ds_shape):
                pop_num = 0
                for i, in_ir_s in enumerate(in_ir_shape[::]):
                    if in_ir_s == 1:
                        in_ir_shape.pop(i-pop_num)
                        pop_num += 1
                        if len(in_ir_shape) == len(in_ds_shape):
                            break
            else:
                insert_num = 0
                for i, in_ds_s in enumerate(in_ds_shape[::]):
                    if in_ds_s == 1 and i < len(in_ir_shape) and in_ir_shape[i] != 1:
                        in_ir_shape.insert(i, 1)
                        insert_num += 1
                        if len(in_ir_shape) == len(in_ds_shape):
                            break

    if in_ir_shape == in_ds_shape:
        return out_ir_shape

    change_axis = []
    for i, (irs, dss) in enumerate(zip(in_ir_shape, in_ds_shape)):
        if irs != dss:
            change_axis.append(i)

    if len(change_axis) > 1:
        OPT_WARN(f"dynamic shape has more than one change axis, which now does not support.")

    change_axis = change_axis[0]

    in_ps = 0
    ot_ps = 0
    in_begin = 0
    ot_begin = 0

    # [[[in_shape_axis], [out_shape_axis]]] like [[[0, 1], [0, 1, 2]], [[2], [3]]]
    matched_axis = []
    in_mul = in_ir_shape[0]
    ot_mul = out_ir_shape[0]

    while 1:
        if in_mul == ot_mul:
            in_visited = [v for v in range(in_begin, in_ps+1)]
            ot_visited = [v for v in range(ot_begin, ot_ps+1)]
            matched_axis.append([in_visited, ot_visited])
            in_ps += 1
            ot_ps += 1
            in_begin = in_ps
            ot_begin = ot_ps
            if in_ps > len(in_ir_shape) - 1 and ot_ps > len(out_ir_shape) - 1:
                break
            elif in_ps > len(in_ir_shape) - 1 and ot_ps <= len(out_ir_shape) - 1:
                ot_visited += [v for v in range(ot_ps, len(out_ir_shape))]
                break
            elif ot_ps > len(out_ir_shape) - 1 and in_ps <= len(in_ir_shape) - 1:
                in_visited += [v for v in range(in_ps, len(in_ir_shape))]
                break

            in_mul = in_ir_shape[in_ps]
            ot_mul = out_ir_shape[ot_ps]
        elif in_mul < ot_mul:
            in_ps += 1
            in_mul *= in_ir_shape[in_ps]
        else:  # in_mul > ot_mul:
            ot_ps += 1
            ot_mul *= out_ir_shape[ot_ps]

    out_ds_shape = [-1] * len(out_ir_shape)
    for match_a in matched_axis:
        if len(match_a[0]) > 1 and len(match_a[1]) == 1:
            mul = 1
            for axis in match_a[0]:
                mul *= in_ds_shape[axis]
            out_ds_shape[match_a[1][0]] = mul
        elif len(match_a[1]) > 1 and len(match_a[0]) == 1:
            if change_axis in match_a[0]:
                mul = 1
                for i in match_a[1]:
                    if out_ir_shape[i] == 1:
                        out_ds_shape[i] = 1
                        mul *= 1
                        continue
                    if in_ds_shape[match_a[0][0]] % out_ir_shape[i] == 0:
                        out_ds_shape[i] = out_ir_shape[i]
            else:
                for i in match_a[1]:
                    out_ds_shape[i] = out_ir_shape[i]
        elif len(match_a[0]) == 1 and len(match_a[1]) == 1:
            out_ds_shape[match_a[1][0]] = in_ds_shape[match_a[0][0]]
        else:  # mulit-input axis <--> multi-output axis
            # [[[0, 1], [0, 1, 2]], [[2], [3]]]
            if change_axis not in match_a[0]:
                for i in match_a[1]:
                    out_ds_shape[i] = out_ir_shape[i]
            else:
                # [[[0, 1], [0, 1, 2]], [[2], [3]]]
                mul = 1
                in_idx = len(match_a[0]) - 1
                for i in match_a[1][::-1]:
                    mul *= out_ir_shape[i]
                    if mul <= in_ir_shape[in_idx]:
                        out_ds_shape[i] = out_ir_shape[i]
                    else:
                        in_idx -= 1
                        mul = 1
                        if in_idx < 0:
                            break
                        continue
    return out_ds_shape


@op_register(OpType.Reshape)
def reshape(self, *args):
    inp = self.inputs[0].betensor.clone()
    out = self.outputs[0]
    try:
        shape = get_ds_shape(list(self.inputs[0].ir_shape), list(out.ir_shape), list(inp.shape))
        out.betensor = torch.reshape(inp, shape)
        return out.betensor
    except Exception as e:
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
