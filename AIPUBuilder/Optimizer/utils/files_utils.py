# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

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
