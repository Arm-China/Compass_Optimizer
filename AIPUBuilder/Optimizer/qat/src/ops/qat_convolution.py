# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


import torch
from torch import nn
from torch.nn.parameter import Parameter
from AIPUBuilder import ops
from .qat_base_operator import QBaseOperator, check_args
from ..qinfo import QuantStage, QScheme
from ..qatlogger import QAT_ERROR, QAT_WARN, QAT_INFO
from ..qatregister import register_operator
from ..config import QATConfig


@register_operator()
class QConvolution2D(QBaseOperator):
    def __init__(self,
                 name,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode="zeros",
                 act_function=None,
                 dtype=None,
                 conv_node=None,
                 ) -> None:
        super().__init__(dtype)

        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.dtype = dtype
        self.conv_node = conv_node

        self.bias = Parameter(torch.zeros(self.out_channels,)) if bias else None
        self.weight = Parameter(torch.rand((self.out_channels, self.in_channels, *self.kernel_size)))
        self.act_function = act_function

        self.activation_qinfo = QATConfig.get('activation_qinfo')
        self.weight_qinfo = QATConfig.get('weight_qinfo')
        self.bias_qinfo = QATConfig.get('bias_qinfo')

        if self.groups > 1:
            self.weight_qinfo.group = self.groups
            if not QScheme.is_perchannel(self.weight_qinfo.qscheme):
                perchannel_scheme = QScheme.str(self.weight_qinfo.qscheme).replace('tensor', 'channel')
                self.weight_qinfo.qscheme = QScheme.str_to_qscheme(perchannel_scheme)

    @check_args
    def forward(self, inputs, iweight=None, ibias=None):
        if iweight is None and ibias is None:
            # weight = self.fake_quant(self.weight, self.weight_qinfo)
            output_scale = self.activation_qinfo.scale
            if False and QScheme.is_perchannel(self.weight_qinfo.qscheme) and not torch.equal(output_scale, torch.ones_like(output_scale)):
                weight = self.fake_quant_weight_with_unified(self.weight,
                                                             self.weight_qinfo,
                                                             output_scale,
                                                             torch.tensor(1.0).to(self.weight_qinfo.scale.device))
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
        elif iweight is not None and ibias is not None:
            weight = iweight
            bias = ibias
        else:
            weight = iweight if iweight is not None else self.fake_quant(self.weight, self.weight_qinfo)
            bias = ibias if ibias is not None else self.fake_quant(self.bias, self.bias_qinfo)

        outputs = torch.nn.functional.conv2d(input=inputs,
                                             weight=weight,
                                             bias=bias.float(),
                                             stride=self.stride,
                                             padding=self.padding,
                                             dilation=self.dilation,
                                             groups=self.groups)
        if self.act_function:
            outputs = self.act_function(outputs)
        outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def serialize(self, inputs):
        from AIPUBuilder import ops
        from AIPUBuilder.core import Tensor, Dtype
        with_act_dict = {
            torch.nn.ReLU: "RELU",
            torch.nn.ReLU6: "RELU6",
            torch.nn.LeakyReLU: "LEAKYRELU",
            torch.nn.PReLU: "PRELU",
            None: "NONE"
        }
        transpose_out = ops.transpose(inputs, perm=[0, 2, 3, 1])
        with_activation = with_act_dict[type(self.act_function)] if self.act_function is not None else "NONE"

        weight = Tensor(self.name + "_weight", self.weight.cpu().numpy().astype('float32').transpose([0, 2, 3, 1]))
        bias = Tensor(self.name + "_bias", self.bias.cpu().numpy().astype('float32'))
        if self.padding[0] != self.padding[1]:
            QAT_ERROR(f"now qconvolution only supports padding[0] == padding[1], but now is {self.padding}")
        out_q = self.get_quantization(self.activation_qinfo)
        if out_q is not None:
            self.update_tensor_quantization(weight, self.weight_qinfo)
            self.update_tensor_quantization(bias, self.bias_qinfo)

        conv_out = ops.conv2d(transpose_out,
                              weight,
                              bias,
                              self.stride,
                              (self.padding[0], self.padding[1],
                               self.padding[0], self.padding[1]),
                              self.dilation,
                              with_activation,
                              quantization=out_q)
        conv_trans_out = ops.transpose(conv_out, perm=[0, 3, 1, 2])
        conv_trans_out.op.name = self.name
        return conv_trans_out
