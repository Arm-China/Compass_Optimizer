# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import OPT_DEBUG, OPT_INFO, OPT_ERROR

register_optype('GetValidCount')


@op_register(OpType.GetValidCount)
def get_valid_count(self, *args):
    cls_prob = self.inputs[0].betensor

    batch_size = cls_prob.shape[0]
    num_anchors = cls_prob.shape[1]
    box_data_length = cls_prob.shape[2]

    valid_count = -torch.ones([batch_size], device=cls_prob.device)
    out_tensor = -torch.ones([batch_size, num_anchors, box_data_length], device=cls_prob.device)
    out_indices = -torch.ones([batch_size, num_anchors], device=cls_prob.device)

    score_threshold = self.get_param('score_threshold_value') if self.quantized else self.get_param('score_threshold')
    score_index = torch.tensor(int(self.get_param('score_index')), device=cls_prob.device)
    id_index = torch.tensor(int(self.get_param('id_index', optional=True, default_value=-1)),  device=cls_prob.device)

    if id_index >= 0:
        id_data = torch.index_select(cls_prob, dim=2, index=id_index)
    else:
        id_data = torch.ones([batch_size, num_anchors], device=cls_prob.device)
    initial_index = torch.arange(0, num_anchors, device=cls_prob.device).long()
    score_data = torch.index_select(cls_prob, dim=2, index=score_index)

    for b in range(batch_size):
        batch_data = cls_prob[b]
        batch_score_data = score_data[b].reshape(-1,)
        batch_id_data = id_data[b].reshape(-1,)
        score_mask = batch_score_data > score_threshold
        index_mask = batch_id_data >= 0
        valid_mask = torch.bitwise_and(score_mask, index_mask)
        valid_index = initial_index[valid_mask]
        valid_data = torch.index_select(batch_data, dim=0, index=valid_index)
        valid_num = valid_data.shape[0]
        valid_count[b] = valid_num
        out_tensor[b, :valid_num, :] = valid_data
        out_indices[b, :valid_num] = valid_index

    self.outputs[0].betensor = valid_count
    self.outputs[1].betensor = out_tensor
    self.outputs[2].betensor = out_indices

    return (valid_count, out_tensor, out_indices)


@quant_register(OpType.GetValidCount)
def get_valid_count_quantize(self, *args):
    inp = self.inputs[:]
    out = self.outputs[:]

    score_thresh = self.params.pop('score_threshold')
    score_thresh = linear_quantize_clip(score_thresh, inp[0].scale, inp[0].zerop, inp[0].qmin, inp[0].qmax).item()
    self.params['score_threshold_value'] = int(score_thresh)
    self.params['score_threshold_type'] = inp[0].dtype

    # set dtpye and qbits
    out[0].scale, out[0].zerop, out[0].dtype, out[0].qbits = 1, 0, Dtype.INT16, 16
    out[0].qinvariant = True
    out[1].scale, out[1].zerop, out[1].dtype, out[1].qbits = inp[0].scale, inp[0].zerop, inp[0].dtype, inp[0].qbits
    out[1].qinvariant = inp[0].qinvariant
    out[2].scale, out[2].zerop, out[2].dtype, out[2].qbits = 1, 0, Dtype.INT16, 16
    out[2].qinvariant = True
