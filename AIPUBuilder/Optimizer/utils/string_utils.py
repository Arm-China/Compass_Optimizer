# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.logger import *


def list_any_to_str(s):
    if isinstance(s, list):
        lstr = '['
        for x in s:
            lstr += str(x) + ','
        if len(lstr) > 1:
            lstr = lstr[:-1] + ']'
        else:
            lstr += ']'
        return lstr
    else:
        return str(s)


def timestamp_string():
    from datetime import datetime
    import random
    return '_' + str(datetime.timestamp(datetime.now())).replace('.', '_') + '_' + str(random.random()).replace('.', '_') + '_'
