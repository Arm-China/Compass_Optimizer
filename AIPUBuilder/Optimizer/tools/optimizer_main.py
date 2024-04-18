# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import sys
from AIPUBuilder.Optimizer.plugins import *
try:
    from AIPUBuilder.Optimizer.plugins_internal import *
except:
    pass
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.config import *
from AIPUBuilder.Optimizer.optmaster import *
from AIPUBuilder.Optimizer.logger import OPT_START, OPT_END


def OPT_WORK(argv):
    graph = QuantizeGraph.parse(argv.graph, argv.bin)
    optimizer = OptMaster(graph, argv)
    report = optimizer()
    return report


def main():
    try:
        traverse_opt_plugins()
        argv = arg_parser(metric_dict=QUANTIZE_METRIC_DICT, dataset_dict=QUANTIZE_DATASET_DICT)
        if isinstance(argv, bool):
            return 0 if argv is True else 1  # return 0/1 value for tvm calling the optimizer

        OPT_START(argv)
        report = OPT_WORK(argv)
        OPT_END(report)
        return 0
    except Exception as e:
        OPT_END()
        raise e


if __name__ == '__main__':

    ret = main()
    sys.exit(ret)
