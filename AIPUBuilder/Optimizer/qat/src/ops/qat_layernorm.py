# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


import torch
import math
from torch.nn.parameter import Parameter
from functools import reduce
from AIPUBuilder import ops
from ..qinfo import QuantStage, QInfo, QScheme
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QLayerNorm(QBaseOperator):
    def __init__(self,
                 name,
                 normalized_shape,
                 eps=0.000001,
                 group=1,
                 bias=True,
                 dtype=None) -> None:
        super().__init__(dtype)

        self.name = name
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.axis = [0 - ax - 1 for ax in range(len(self.normalized_shape))]
        self.group = 1

        self.bias = Parameter(torch.zeros(self.normalized_shape)) if bias else None
        self.weight = Parameter(torch.ones(self.normalized_shape))

        self.activation_qinfo = QATConfig.get('activation_qinfo')
        self.weight_qinfo = QATConfig.get('weight_qinfo')
        self.bias_qinfo = QATConfig.get('bias_qinfo')

        self.normed_qinfo = QInfo().clone(self.activation_qinfo)
        self.normed_qinfo.qscheme = QScheme.to_symmetric(self.normed_qinfo.qscheme)

    def _norm_quantize(self):
        if not self.normed_qinfo.is_quantized:
            normalized_shape = self.normalized_shape
            if not isinstance(self.normalized_shape, (tuple, list)):
                normalized_shape = [self.normalized_shape, ]
            shape_prod = reduce(lambda x, y: x*y, normalized_shape)
            sqrt_n = math.sqrt(shape_prod / self.group)
            alpha = 1.0
            adj = 0
            if self.normed_qinfo.bits <= 8 and sqrt_n > 127.0:
                adj = math.ceil(math.log2(127.5 / sqrt_n / 4))
                alpha = math.pow(2, adj)
            sqrt_n *= alpha
            self.normed_qinfo.mins = 0 - sqrt_n
            self.normed_qinfo.maxs = sqrt_n
            self.linear_affine(self.normed_qinfo)

    @check_args
    def forward(self, inputs, iweight=None, ibias=None):
        if iweight is not None and ibias is not None:
            weight, bias = iweight, ibias
        else:
            weight = self.fake_quant(self.weight, self.weight_qinfo)
            if self.quant_stage in [QuantStage.QAT, QuantStage.INFER]:
                self._norm_quantize()
                from AIPUBuilder.Optimizer.utils import bits2range
                self.bias_qinfo.scale = self.normed_qinfo.scale * self.weight_qinfo.scale
                self.bias_qinfo.zerop = torch.zeros_like(self.bias_qinfo.scale)
                self.bias_qinfo.qmin, self.bias_qinfo.qmax = bits2range(self.bias_qinfo.bits, True)
                bias = self._linear_quantize_dequantize(self.bias, self.bias_qinfo)
            else:
                bias = self.fake_quant(self.bias, self.bias_qinfo)
                # bias = self.bias
        outputs = torch.nn.functional.layer_norm(inputs,
                                                 self.normalized_shape,
                                                 weight=weight,
                                                 bias=bias.float(),
                                                 eps=self.eps)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def serialize(self, inputs):
        from AIPUBuilder import ops
        from AIPUBuilder.core import Tensor, Dtype

        weight = Tensor(self.name + "_weight", self.weight.cpu().numpy().astype('float32'))
        bias = Tensor(self.name + "_bias", self.bias.cpu().numpy().astype('float32'))
        out_q = self.get_quantization(self.activation_qinfo)
        if out_q is not None:
            self.update_tensor_quantization(weight, self.weight_qinfo)
            self.update_tensor_quantization(bias, self.bias_qinfo)

        out = ops.layer_norm(inputs,
                             weight,
                             bias,
                             axis=self.axis,
                             epsilon=self.eps,
                             quantization=out_q)
        out.op.name = self.name
        return out
