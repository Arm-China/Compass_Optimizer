# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

__all__ = ['make_path', 'make_dir_path']


def make_path(path):
    dpath = os.path.dirname(path)
    if not os.path.exists(dpath):
        os.makedirs(dpath)
    return path


def make_dir_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
