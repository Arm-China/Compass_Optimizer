# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import sys
from AIPUBuilder.Optimizer.plugins import *
try:
    from AIPUBuilder.Optimizer.plugins_internal import *
except:
    pass

from AIPUBuilder.Optimizer.framework import (traverse_opt_plugins,
                                             QUANTIZE_DATASET_DICT,
                                             QUANTIZE_METRIC_DICT)
from AIPUBuilder.Optimizer.config import arg_parser
from src import AIPUQATMaster
from src.qatlogger import QAT_INFO


def main():
    try:
        traverse_opt_plugins()
        args = arg_parser(metric_dict=QUANTIZE_METRIC_DICT,
                          dataset_dict=QUANTIZE_DATASET_DICT)  # pylint: disable=undefined-variable
        if isinstance(args, bool):
            return 0 if args else 1

        qat_master = AIPUQATMaster(args)
        qat_master.run()
        QAT_INFO(f"running QAT Done.")
    except Exception as e:
        raise e


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
