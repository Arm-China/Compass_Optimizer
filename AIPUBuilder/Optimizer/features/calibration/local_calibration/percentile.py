# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


def percentile_calibration(t, *args):
    cstrategy = args[0]
    try:
        p = float(cstrategy[:-10])
    except:
        p = 1.0
    t.min = t.extrema_min * p
    t.max = t.extrema_max * p
    if t.extrema_min_key_axis is not None:
        t.min_key_axis = t.extrema_min_key_axis * p
        t.max_key_axis = t.extrema_max_key_axis * p
