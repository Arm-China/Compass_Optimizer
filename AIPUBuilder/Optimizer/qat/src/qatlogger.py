# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.e


from AIPUBuilder.Optimizer.logger import OPT_INFO, OPT_ERROR, OPT_WARN, OPT_DEBUG, OPT_FATAL


def QAT_INFO(*args, **kwargs):
    OPT_INFO(*args, **kwargs, prefix_header='[QAT]')


def QAT_ERROR(*args, **kwargs):
    OPT_ERROR(*args, **kwargs, prefix_header='[QAT]')


def QAT_WARN(*args, **kwargs):
    OPT_WARN(*args, **kwargs, prefix_header='[QAT]')


def QAT_DEBUG(*args, **kwargs):
    OPT_DEBUG(*args, **kwargs, prefix_header='[QAT]')


def QAT_FATAL(*args, **kwargs):
    OPT_FATAL(*args, **kwargs, prefix_header='[QAT]')
