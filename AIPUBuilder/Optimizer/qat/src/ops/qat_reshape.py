# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QReshape(QBaseOperator):
    def __init__(self,
                 shape=None,
                 start_dim=None,
                 end_dim=None,
                 dtype=None,
                 name='') -> None:
        super().__init__(dtype)
        self._use_input_QConfig = True
        self.shape = shape
        self.start_dim = start_dim if start_dim is not None else 0
        self.end_dim = end_dim if end_dim is not None else -1
        self._is_flatten = True if start_dim is not None or end_dim is not None else False
        self.activation_qinfo = QATConfig.get('activation_qinfo')
        self.name = name

    @check_args
    def forward(self, inputs, *args):
        if self._is_flatten:
            outputs = torch.flatten(inputs, self.start_dim, self.end_dim)
        else:
            if len(args):
                outputs = torch.reshape(inputs, args)
                self.shape = list(args)
            else:
                outputs = torch.reshape(inputs, self.shape)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def serialize(self, input):
        from AIPUBuilder import ops

        if self.shape is None:
            dim = len(input.shape)
            self.shape = []
            sd = self.start_dim if self.start_dim >= 0 else (self.start_dim + dim)
            ed = self.end_dim if self.end_dim >= 0 else (self.end_dim + dim)
            in_shape = input.shape
            has_append = False
            f = 1
            for d in range(dim):
                if d < sd or d > ed:
                    self.shape.append(in_shape[d])
                else:
                    f *= in_shape[d]
                    if not has_append:
                        self.shape.append(-1)
                        has_append = True
            self.shape[self.shape.index(-1)] = f
        else:
            # TODO using in_shape and self.shape to judge the self.shape is valid, like self.shape = [10,224,224,3] and in_shape=[1,224,224,3]
            pass

        out = ops.reshape(input, self.shape)
        out.quantization = input.quantization
        return out
