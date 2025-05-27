# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

def convert2tuple(input):
    '''For DNN parameters conversion'''
    return input if isinstance(input, tuple) else (input, input)
