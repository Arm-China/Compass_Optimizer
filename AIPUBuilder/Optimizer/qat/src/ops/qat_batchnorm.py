# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


import torch
from torch import nn
from torch.nn.parameter import Parameter
from .qat_base_operator import QBaseOperator
from ..qatregister import register_operator
from ..config import QATConfig


@register_operator()
class QBatchNorm(QBaseOperator):
    def __init__(self,
                 num_features,
                 name="",
                 dim=-1,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 device=None,
                 dtype=None) -> None:
        super().__init__(dtype, name)

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.dtype = dtype

        self.dim = dim

        '''FusedBN'''
        # self.bias = Parameter(torch.zeros(self.num_features,))
        # self.weight = Parameter(torch.rand((self.num_features,)))

        self.register_buffer('bias', torch.zeros(self.num_features,))
        self.register_buffer('weight', torch.ones(self.num_features,))

        self.activation_qinfo = QATConfig.get('activation_qinfo')
        self.weight_qinfo = QATConfig.get('weight_qinfo')
        self.weight_qinfo.bits = 16
        self.bias_qinfo = QATConfig.get('bias_qinfo')

    def forward(self, inputs):
        weight = self.fake_quant(self.weight, self.weight_qinfo)
        bias = self.fake_quant(self.bias, self.bias_qinfo)
        outputs = inputs * weight + bias
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def serialize(self, inputs):
        from AIPUBuilder import ops
        from AIPUBuilder.core import Tensor
        import numpy as np
        w = self.weight.cpu().numpy().astype('float32')
        if self.weight.numel() == 1 and inputs.shape[self.dim] > 1:
            w = np.tile(w, inputs.shape[self.dim])
        b = self.bias.cpu().numpy().astype('float32')
        if self.bias.numel() == 1 and inputs.shape[self.dim] > 1:
            b = np.tile(b, inputs.shape[self.dim])

        weight = Tensor(self.name + "_weight", w)
        bias = Tensor(self.name + "_bias", b)

        out_q = self.get_quantization(self.activation_qinfo)
        if out_q is not None:
            self.update_tensor_quantization(weight, self.weight_qinfo)
            self.update_tensor_quantization(bias, self.bias_qinfo)

        return ops.batch_norm(inputs, weight, bias, axis=self.dim, quantization=out_q)
