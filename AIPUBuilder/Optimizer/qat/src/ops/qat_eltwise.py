# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from torch import nn
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


class QElementwise(QBaseOperator):
    def __init__(self, name, dtype=None, act_function=nn.ReLU) -> None:
        super().__init__(dtype=dtype, name=name)
        self.dtype = dtype
        self.act_function = act_function
        self.activation_qinfo = QATConfig.get('activation_qinfo')
        self.method = 'ADD'

    def serialize(self, input1, input2):
        from AIPUBuilder import ops

        with_activation = "NONE"
        if self.act_function:
            with_act_dict = {
                torch.nn.ReLU: "RELU",
                torch.nn.ReLU6: "RELU6",
                torch.nn.LeakyReLU: "LEAKYRELU",
                torch.nn.PReLU: "PRELU",
            }
            with_activation = with_act_dict[type(self.act_function)]

        method_d = {
            "ADD": ops.add,
            "SUB": ops.sub,
            "MUL": ops.mul,
        }
        operator = method_d[self.method]
        out_q = self.get_quantization(self.activation_qinfo)
        out = operator(input1, input2, activation=with_activation, quantization=out_q)
        return out


@register_operator()
class QElementwiseAdd(QElementwise):
    def __init__(self,
                 name,
                 dtype=None,
                 act_function=nn.ReLU()) -> None:
        super().__init__(name=name, dtype=dtype, act_function=act_function)
        self.method = "ADD"

    @check_args
    def forward(self, input1, input2):
        outputs = input1 + input2
        if self.act_function:
            outputs = self.act_function(outputs)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs


@register_operator()
class QElementwiseMul(QElementwise):
    def __init__(self,
                 name,
                 dtype=None,
                 act_function=nn.ReLU()) -> None:
        super().__init__(name=name, dtype=dtype, act_function=act_function)
        self.method = "MUL"

    @check_args
    def forward(self, input1, input2):
        outputs = input1 * input2
        if self.act_function:
            outputs = self.act_function(outputs)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs
