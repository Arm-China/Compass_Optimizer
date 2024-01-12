# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

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


def string_to_base_type(s: str):
    import re
    opt_v = s.strip()
    if opt_v.upper() == "FALSE":
        opt_v = False
    elif opt_v.upper() == "TRUE":
        opt_v = True
    elif re.findall('^[-+]?\d+$', opt_v):
        opt_v = int(opt_v)
    elif re.findall('^[-+]?[0-9]+\.?[0-9]*$', opt_v):
        opt_v = float(opt_v)
    return opt_v
