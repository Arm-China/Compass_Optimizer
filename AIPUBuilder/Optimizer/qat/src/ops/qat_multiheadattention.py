# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from ..qatlogger import QAT_ERROR, QAT_WARN, QAT_INFO
from ..qatregister import register_operator
from ..config import QATConfig
from .qat_base_operator import QBaseOperator, check_args


@register_operator()
class QMultiHeadAttention(QBaseOperator):
    def __init__(self,
                 name,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False,
                 kdim=None,
                 vdim=None,
                 batch_first=False,
                 need_weights=False,
                 dtype=None) -> None:
        super().__init__(dtype)

        self.name = name
        self.self_attention = torch.nn.MultiheadAttention(embed_dim=embed_dim,
                                                          num_heads=num_heads,
                                                          dropout=dropout,
                                                          bias=bias,
                                                          add_bias_kv=add_bias_kv,
                                                          add_zero_attn=add_zero_attn,
                                                          kdim=kdim,
                                                          vdim=vdim,
                                                          batch_first=batch_first,
                                                          dtype=None)
        self.need_weights = need_weights
        self.activation_qinfo = QATConfig.get('activation_qinfo')
        self.seq_len = None

    def _local_forward(self, *args, **kwargs):
        pass

    def _torch_forward(self, *args, **kwargs):
        outputs = self.self_attention.forward(*args, **kwargs)  # (attn_output, attn_output_weights)
        return outputs

    # def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
    @check_args
    def forward(self, *args, **kwargs):
        self.seq_len = args[0].shape[1]
        outputs = self._torch_forward(*args, **kwargs)
        # outputs = self.self_attention.forward(*inputs, need_weights=self.need_weights) # (attn_output, attn_output_weights)
        # outputs = self.self_attention.forward(*args, **kwargs)  # (attn_output, attn_output_weights)
        outputs = list(outputs)
        if len(outputs) == 2:
            outputs[0] = self.fake_quant(outputs[0], self.activation_qinfo)
        else:
            outputs = self.fake_quant(outputs, self.activation_qinfo)
        return outputs

    def serialize(self, input, *args):
        from AIPUBuilder.core import Tensor, Node, Graph, register_optype, OpType, Dtype, TensorShape
        from AIPUBuilder import ops
        register_optype('MultiheadAttention')

        mha = Node(f"{self.name}", OpType.MultiheadAttention)
        if isinstance(input, (list, tuple)):
            for i, inp in enumerate(input):
                mha.add_input(inp, i)
        else:
            mha.add_input(input)
        o1 = Tensor(f"{self.name}_output0", input.shape, input.dtype)
        o2 = Tensor(f"{self.name}_output1", input.shape, input.dtype)
        mha.add_output(o1)
        mha.add_output(o2)

        if isinstance(input, (list, tuple)):
            input[0].op.graph.add_node(mha)
        else:
            input.op.graph.add_node(mha)
        return (o1, o2)
