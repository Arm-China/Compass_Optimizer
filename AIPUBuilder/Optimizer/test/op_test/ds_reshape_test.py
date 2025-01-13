# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.
import torch


def get_ds_shape(in_ir_shape, out_ir_shape, in_ds_shape):
    if len(in_ir_shape) != len(in_ds_shape):
        OPT_ERROR(f"In Reshape Op: the length of runtime input shape(={len(in_ds_shape)}) != "
                  f"the length of input ir_shape(={len(in_ir_shape)})")

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
            if in_ps > len(in_ir_shape) - 1 or ot_ps > len(out_ir_shape) - 1:
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
    print(matched_axis)
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
                # if len(match_a[0]) < len(match_a[1]):
                #     mul = 1
                #     in_idx = len(match_a[0]) - 1
                #     for i in match_a[1][::-1]:
                #         mul *= out_ir_shape[i]
                #         if mul <= in_ir_shape[in_idx]:
                #             out_ds_shape[i] = out_ir_shape[i]
                #         else:
                #             in_idx -= 1
                #             mul = 1
                #             if in_idx < 0:
                #                 break
                #             continue

    return out_ds_shape


cases = [
    # [[1, 6, 2500, 256], [15000, 256], [1, 6, 604, 256], [3624, 256]],
    # [[1, 6, 2500, 256], [6, 2500, 256], [1, 6, 604, 256], [6, 604, 256]],
    # [[15000, 64], [6, 2500, 8, 8], [6*604, 64], [6, 604, 8, 8]],
    # [[6, 2500, 8, 8], [6, 2500, 8, 1, 8], [6 , 604, 8, 8], [6, 604, 8, 1, 8]],
    # [[6, 8, 2500, 1, 8], [48, 1, 2500, 8], [6, 8, 604, 1, 8], [48, 1, 604, 8]],
    # [[48, 32, 2500, 1], [1, 6, 256, 2500], [48, 32, 604, 1], [1, 6, 256, 604]],
    # [[1, 2500, 256], [2500, 256], [1, 604, 256], [604, 256]],
    # [[15000, 128], [6, 20000, 8, 2], [3624, 128], [6, 604 * 8, 8, 2]],
    # [[6, 20000, 8, 2], [6, 2500, 8, 1, 2, 4, 2], [6, 604 * 8, 8, 2], [6, 604, 8, 1, 2, 4, 2]],
    [[6, 20000, 8, 2], [15000, 128], [6, 604*8, 8, 2], [3624, 128]],

]

for case in cases:
    out = get_ds_shape(case[0], case[1], case[2])
    print(out)
    # assert out[1] == case[3], f"{case}"
    a = torch.rand(case[2])
    b = a.reshape(case[3])
    c = a.reshape(out)
    assert torch.equal(b, c)
