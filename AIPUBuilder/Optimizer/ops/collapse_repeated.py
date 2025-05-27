# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *

register_optype('CollapseRepeated')

# IR
# layer_id=2
# layer_name=collapse_repeated
# layer_type=CollapseRepeated
# layer_bottom=[input0,input1]
# layer_bottom_shape=[[4,1000],[4]]
# layer_bottom_type=[int16,uint32]
# layer_top=[output1,output2]
# layer_top_shape=[[4,1000],[1]]
# layer_top_type=[int16,uint32]


@op_register(OpType.CollapseRepeated)
def collapse_forward(self, *args):
    if len(self.inputs) != 2:
        OPT_FATAL("CollapseRepeated(layer_name=%s) input number must be 2, but now input number is %d" %
                  (self.name, len(self.inputs)))
    labels = self.inputs[0].betensor
    seq_length = self.inputs[1].betensor.long()
    if labels.ndim != 2:
        OPT_FATAL("CollapseRepeated(layer_name=%s) 0-input dim must be 2, but now is %d" % (self.name, labels.ndim))
    if seq_length.ndim != 1:
        OPT_FATAL("CollapseRepeated(layer_name=%s) 1-input dim must be 1, but now is %d" % (self.name, seq_length.ndim))
    if labels.shape[0] != seq_length.shape[0]:
        OPT_FATAL("CollapseRepeated(layer_name=%s) two input batch dim must be equal" % (self.name))

    batch = labels.shape[0]
    maxlen = labels.shape[1]
    out_newlen = self.outputs[0].ir_shape[1]
    label_ones = torch.ones_like(labels[:, :1]).bool()
    label_xor = (labels[:, 1:] != labels[:, :-1])
    label_mask = torch.cat((label_ones, label_xor), dim=1)

    seq_mask = torch.zeros((batch, maxlen), device=labels.device).bool()
    for b in range(batch):
        if seq_length[b] > 0:
            seq_mask[b, :seq_length[b]] = True
    label_mask = torch.bitwise_and(seq_mask, label_mask)

    new_seq_len = torch.sum(label_mask.int(), dim=1, keepdim=False)
    new_maxlen = torch.max(new_seq_len).int().item()
    idx_mask = torch.zeros((batch, new_maxlen), device=labels.device).bool()
    for b in range(batch):
        idx_mask[b, :new_seq_len[b]] = True

    padding_value = -self.inputs[0].zerop[0] if self.quantized else 0
    padding_diff = (0, out_newlen - new_maxlen)

    flat_labels = torch.flatten(labels)
    flat_label_mask = torch.flatten(label_mask)
    flat_idx_mask = torch.flatten(idx_mask)
    idx = torch.arange(idx_mask.numel(), device=labels.device).long()
    output = (torch.zeros_like(flat_idx_mask, dtype=labels.dtype, device=labels.device) + torch.tensor(padding_value, dtype=labels.dtype))\
        .scatter_(0, idx[flat_idx_mask], flat_labels[flat_label_mask])
    output = torch.reshape(output, (batch, new_maxlen))

    self.outputs[0].betensor = torch.nn.functional.pad(output, padding_diff, "constant", padding_value)
    self.outputs[1].betensor = new_seq_len

    return (self.outputs[0].betensor, self.outputs[1].betensor)


@quant_register(OpType.CollapseRepeated)
def collapse_quantize(self, *args):
    for idx, inp in enumerate(self.inputs):
        self.outputs[idx].dtype = self.inputs[idx].dtype
        self.outputs[idx].scale = self.inputs[idx].scale
        self.outputs[idx].zerop = self.inputs[idx].zerop
        self.outputs[idx].qbits = self.inputs[idx].qbits
        self.outputs[idx].qinvariant = self.inputs[idx].qinvariant
