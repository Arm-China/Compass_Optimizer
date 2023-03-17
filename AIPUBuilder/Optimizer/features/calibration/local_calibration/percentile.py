# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


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
