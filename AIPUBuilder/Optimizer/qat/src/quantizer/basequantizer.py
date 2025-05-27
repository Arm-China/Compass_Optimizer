# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


import torch


class QATBaseQuantizer(object):

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.ir_mode = ""

    def forward(self, model, input):
        output = model(*input)
        if isinstance(output, torch.Tensor):
            output = [output]
        return output
