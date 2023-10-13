# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


def mean_calibration(t, *args):
    t.min = t.running_min
    t.max = t.running_max
    if t.running_min_key_axis is not None:
        t.min_key_axis = t.running_min_key_axis
        t.max_key_axis = t.running_max_key_axis
