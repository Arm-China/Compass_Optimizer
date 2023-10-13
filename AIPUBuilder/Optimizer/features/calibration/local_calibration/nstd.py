# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

import torch


def nstd_calibration(t, *args):
    # n = int(cstrategy[:-3])
    n = int(args[0][:-3])
    t.min = max(t.running_min, t.running_mean - n * t.running_std)
    t.max = min(t.running_max, t.running_mean + n * t.running_std)
    if t.running_mean_key_axis is not None:
        t.min_key_axis = torch.max(t.running_min_key_axis, t.running_mean_key_axis - n * t.running_std_key_axis)
        t.max_key_axis = torch.min(t.running_max_key_axis, t.running_mean_key_axis + n * t.running_std_key_axis)
