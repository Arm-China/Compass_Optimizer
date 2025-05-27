# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from torch.nn.parameter import Parameter
from ..qatregister import register_operator
from ..config import QATConfig
from ..qinfo import QuantStage, QScheme
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QFullyConnected(QBaseOperator):
    def __init__(self,
                 name,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 act_function=None) -> None:
        super().__init__(dtype)

        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self._output_shape = None

        self.bias = Parameter(torch.zeros(self.out_features,)) if bias else None
        self.weight = Parameter(torch.rand((self.out_features, self.in_features)))
        self.act_function = act_function

        self.activation_qinfo = QATConfig.get('activation_qinfo')
        self.weight_qinfo = QATConfig.get('weight_qinfo')
        self.bias_qinfo = QATConfig.get('bias_qinfo')

    @check_args
    def forward(self, inputs, iweight=None, ibias=None):
        if iweight is not None and ibias is not None:
            weight, bias = iweight, ibias
        else:
            weight = self.fake_quant(self.weight, self.weight_qinfo)
            if self.quant_stage in [QuantStage.QAT, QuantStage.INFER] and len(self.prev_modules) and hasattr(self.prev_modules[0], 'activation_qinfo'):
                from AIPUBuilder.Optimizer.utils import bits2range
                input_activation_info = self.prev_modules[0].activation_qinfo
                self.bias_qinfo.scale = input_activation_info.scale * self.weight_qinfo.scale
                self.bias_qinfo.zerop = torch.zeros_like(self.bias_qinfo.scale)
                self.bias_qinfo.qmin, self.bias_qinfo.qmax = bits2range(self.bias_qinfo.bits, True)
                bias = self._linear_quantize_dequantize(self.bias, self.bias_qinfo)
            else:
                bias = self.fake_quant(self.bias, self.bias_qinfo)
                # bias = self.bias
        outputs = torch.nn.functional.linear(input=inputs,
                                             weight=weight,
                                             bias=bias)
        if self.act_function:
            outputs = self.act_function(outputs)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        self._output_shape = outputs.shape
        return outputs

    def serialize(self, inputs):
        from AIPUBuilder.core import Dtype, Tensor
        from AIPUBuilder import ops
        weight = Tensor(self.name + "_weight", self.weight.cpu().numpy().astype('float32'))
        bias = Tensor(self.name + "_bais", self.bias.cpu().numpy().astype('float32'))
        out_q = self.get_quantization(self.activation_qinfo)
        if out_q is not None:
            self.update_tensor_quantization(weight, self.weight_qinfo)
            self.update_tensor_quantization(bias, self.bias_qinfo)
        input_reshape = inputs
        if len(inputs.shape) != 2:
            input_reshape = ops.reshape(inputs, [-1, inputs.shape[-1]])
        fc = ops.fully_connected(input_reshape, weight, bias, quantization=out_q)
        if self._output_shape is not None and len(fc.shape) != len(self._output_shape):
            fc_reshape = ops.reshape(fc, list(self._output_shape))
            fc_reshape.op.name = self.name
            return fc_reshape
        fc.op.name = self.name
        return fc
