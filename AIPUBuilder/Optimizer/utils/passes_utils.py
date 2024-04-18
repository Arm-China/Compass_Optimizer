# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import functools
from AIPUBuilder.Optimizer.framework import PyGraph
from AIPUBuilder.Optimizer.logger import OPT_DEBUG

__all__ = ['passes_run', 'PASSES', 'ENABLE_PASSES']

PASSES = dict()
ENABLE_PASSES = dict()


def passes_run(func):
    """
    this decorator is used for enabling or disabling the pass for all nodes, which is setted in cfg file and defaultly
    worked for all nodes. if node has independently flag this decorator does not work.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from AIPUBuilder.Optimizer.config import CfgParser
        flag = len(args) == 2 and isinstance(args[0], PyGraph) and isinstance(args[1], CfgParser)
        flag = flag or (len(args) == 1 and len(kwargs) == 1 and isinstance(args[0], PyGraph)
                        and isinstance(list(kwargs.values())[0], CfgParser))
        if flag:
            hparams = args[1] if len(args) == 2 else list(kwargs.values())[0]
            prefix = 'enable_pass_'
            pass_func_name = f"{prefix}{func.__name__}"
            if not hasattr(hparams, pass_func_name):
                # fixed enable pass, like shrink_pow_exponent
                func(*args, **kwargs)
                OPT_DEBUG(f"now run pass: {func.__name__}")
            elif hasattr(hparams, pass_func_name) and hparams.__getattr__(pass_func_name):
                func(*args, **kwargs)
                OPT_DEBUG(f"now run pass: {func.__name__}")
                if func.__name__ not in ENABLE_PASSES:
                    ENABLE_PASSES.update({func.__name__: func})
    if func.__name__ not in PASSES:
        PASSES.update({func.__name__: wrapper})

    return wrapper
