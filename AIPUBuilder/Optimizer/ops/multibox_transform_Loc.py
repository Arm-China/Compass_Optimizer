# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import OPT_DEBUG, OPT_INFO, OPT_ERROR

register_optype('MultiboxTransformLoc')


def apply_box_deltas(self, boxes, deltas):
    ycenter_a = (boxes[..., 0] + boxes[..., 2]) / 2
    xcenter_a = (boxes[..., 1] + boxes[..., 3]) / 2
    ha = boxes[..., 2] - boxes[..., 0]
    wa = boxes[..., 3] - boxes[..., 1]

    STD = self.get_param('std_div', optional=True, default_value=[10, 10, 5, 5])

    dy = deltas[..., 0] / STD[0]
    dx = deltas[..., 1] / STD[1]
    dh = deltas[..., 2] / STD[2]
    dw = deltas[..., 3] / STD[3]

    # adjust achors size and position
    ycenter = dy * ha + ycenter_a
    xcenter = dx * wa + xcenter_a
    h = torch.exp(dh) * ha
    w = torch.exp(dw) * wa

    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.

    box = torch.stack([ymin, xmin, ymax, xmax], dim=-1)
    # clip in normalized size[0~1.0]
    #  window: (y1, x1, y2, x2)
    box = torch.clip(box, 0, 1.0)
    return box


def apply_box_deltas_q(self, boxes, deltas, boxes_dtype=None, deltas_dtype=None):
    boxes_qbits = dtype2bits(boxes_dtype)
    deltas_qbits = dtype2bits(deltas_dtype)
    scales = self.params["box_scale_value"]
    shifts = self.params["box_shift_value"]
    ycenter = torch.div((boxes[..., 0] + boxes[..., 2]), 2, rounding_mode='floor')
    xcenter = torch.div((boxes[..., 1] + boxes[..., 3]), 2, rounding_mode='floor')
    h = (boxes[..., 2] - boxes[..., 0]).int()
    w = (boxes[..., 3] - boxes[..., 1]).int()

    dy = deltas[..., 0].int()
    dx = deltas[..., 1].int()
    dh = deltas[..., 2].int()
    dw = deltas[..., 3].int()
    scale_ty_out = scales[0]
    shift_ty_out = shifts[0]
    scale_tx_out = scales[1]
    shift_tx_out = shifts[1]
    # sat_bits is to avoid lib calculation saturation,here,so it need (qbits*2+16-32)
    # assume scale is 16bits, max bits 32, currently, only support 8bits and 16bits box
    sat_bits = self.get_param('delta_shift')
    round_shift_y = 1 << (shift_ty_out-sat_bits-1)
    round_shift_x = 1 << (shift_tx_out-sat_bits-1)
    clip_min, clip_max = bits2range(boxes_qbits+deltas_qbits, True)

    h_scale = (h * scale_ty_out) >> sat_bits
    w_scale = (w * scale_tx_out) >> sat_bits
    ycenter_q = torch.clip(((dy * h_scale+round_shift_y) >> (shift_ty_out-sat_bits)) + ycenter, clip_min, clip_max)
    xcenter_q = torch.clip(((dx * w_scale+round_shift_x) >> (shift_tx_out-sat_bits)) + xcenter, clip_min, clip_max)
    th_lut = self.constants['th_lut'].betensor.int()
    tw_lut = self.constants['tw_lut'].betensor.int()
    lut_in_bits = deltas_qbits
    lut_out_bits = 16
    in_is_signed = is_signed(deltas_dtype)
    out_is_signed = True
    scale_th_out = scales[2]
    shift_th_out = shifts[2]
    scale_tw_out = scales[3]
    shift_tw_out = shifts[3]
    clip_min, clip_max = bits2range(boxes_qbits, False)

    lut_dh_scale = (lookup_lut_powerof2(dh, th_lut, lut_in_bits, in_is_signed,
                    lut_out_bits, out_is_signed) * scale_th_out).int() >> 16
    lut_dw_scale = (lookup_lut_powerof2(dw, tw_lut, lut_in_bits, in_is_signed,
                    lut_out_bits, out_is_signed) * scale_tw_out).int() >> 16
    h_half_q = torch.clip((h * lut_dh_scale).int() >> (shift_th_out - 16), clip_min, clip_max)
    w_half_q = torch.clip((w * lut_dw_scale).int() >> (shift_tw_out - 16), clip_min, clip_max)

    scale_anchor = scales[4]
    shift_anchor = shifts[4]
    round_shift = 0
    if shift_anchor > 1:
        round_shift = 1 << (shift_anchor-1)

    ymin_q32 = ((ycenter_q - h_half_q).int() * scale_anchor+round_shift) >> shift_anchor
    xmin_q32 = ((xcenter_q - w_half_q).int() * scale_anchor+round_shift) >> shift_anchor
    ymax_q32 = ((ycenter_q + h_half_q).int() * scale_anchor+round_shift) >> shift_anchor
    xmax_q32 = ((xcenter_q + w_half_q).int() * scale_anchor+round_shift) >> shift_anchor

    box_q32 = torch.stack([ymin_q32, xmin_q32, ymax_q32, xmax_q32], dim=-1)
    box_q16 = torch.clip(box_q32, 0, 32767)  # clip to 0~1.0

    return box_q16


def score_select(self, cls_prob, score_threshold):
    batch_size = cls_prob.shape[0]
    num_anchors = cls_prob.shape[2]
    cls_prob = cls_prob[:, 1:, :]
    value, index = torch.max(cls_prob, dim=1, keepdim=True)
    valid_index = torch.zeros([batch_size, num_anchors], device=cls_prob.device).int()
    valid_batch = torch.zeros([batch_size], device=cls_prob.device).int()
    valid_class_id = torch.zeros([batch_size, num_anchors], device=cls_prob.device).int()
    valid_score = torch.zeros([batch_size, num_anchors], device=cls_prob.device)
    for b in range(batch_size):
        batch_value = value[b].reshape(-1,)
        batch_index = index[b].reshape(-1,)
        valid_mask = batch_value > score_threshold
        initial_box_id = torch.arange(0, num_anchors).long()
        valid_box_id = initial_box_id[valid_mask]
        valid_num = valid_box_id.shape[0]
        valid_index[b, :valid_num] = valid_box_id
        valid_score[b, :valid_num] = batch_value[valid_mask]
        valid_class_id[b, :valid_num] = batch_index[valid_mask]
        valid_batch[b] = valid_num
    return valid_index, valid_class_id, valid_score, valid_batch


def xywh2yxhw(batch_anchor, batch_prod):
    batch_anchor_transformer = torch.zeros_like(batch_anchor)
    batch_prod_transformer = torch.zeros_like(batch_prod)
    batch_anchor_transformer[..., 0] = batch_anchor[..., 1]
    batch_anchor_transformer[..., 1] = batch_anchor[..., 0]
    batch_anchor_transformer[..., 2] = batch_anchor[..., 3]
    batch_anchor_transformer[..., 3] = batch_anchor[..., 2]

    batch_prod_transformer[..., 0] = batch_prod[..., 1]
    batch_prod_transformer[..., 1] = batch_prod[..., 0]
    batch_prod_transformer[..., 2] = batch_prod[..., 3]
    batch_prod_transformer[..., 3] = batch_prod[..., 2]

    return batch_anchor_transformer, batch_prod_transformer


def yxhw2xywh(out_coords):
    ymin = out_coords[..., 0].clone()
    xmin = out_coords[..., 1].clone()
    ymax = out_coords[..., 2].clone()
    xmax = out_coords[..., 3].clone()
    out_coords[..., 0] = xmin
    out_coords[..., 1] = ymin
    out_coords[..., 2] = xmax
    out_coords[..., 3] = ymax

    return out_coords


@op_register(OpType.MultiboxTransformLoc)
def multibox_transform_loc(self, *args):
    cls_prob = self.inputs[0].betensor.float()
    loc_pred = self.inputs[1].betensor.float()
    anchor = self.inputs[2].betensor.float() + (torch.tensor(
        0) if not self.quantized else torch.tensor(self.inputs[2].zerop))

    score_threshold = self.params["score_threshold_value"] if self.quantized else self.params["score_threshold"]

    batch_size = cls_prob.shape[0]
    num_classes = cls_prob.shape[1]
    num_anchors = cls_prob.shape[2]
    loc_pred_reshape = loc_pred.reshape(batch_size, num_anchors, 4)
    outp = -torch.ones([batch_size, num_anchors, 6], device=cls_prob.device)

    valid_index, valid_class_id, valid_score, valid_batch = score_select(self, cls_prob, score_threshold)

    transform_loc = apply_box_deltas_q if self.quantized else apply_box_deltas

    # In order to reuse the boundingbox code later
    if not self.quantized:
        std = [1/x for x in self.params['variance']]
        self.params['std_div'] = [std[1], std[0], std[3], std[2]]
    else:
        loc_pred_reshape[..., :2] += self.inputs[1].zerop

    for b in range(batch_size):
        valid_num = valid_batch[b]
        batch_index = valid_index[b, :valid_num]
        if valid_num != 0:
            batch_anchor = torch.index_select(anchor[0], dim=0, index=batch_index)
            batch_prod = torch.index_select(loc_pred_reshape[b], dim=0, index=batch_index)
            batch_anchor_yxhw, batch_prod_yxhw = xywh2yxhw(batch_anchor, batch_prod)
            out_coords = apply_box_deltas_q(self, batch_anchor_yxhw, batch_prod_yxhw, self.inputs[2].dtype, self.inputs[1].dtype) \
                if self.quantized else apply_box_deltas(self, batch_anchor_yxhw, batch_prod_yxhw)
            out_coords = yxhw2xywh(out_coords)
            outp[b, :valid_num, 0] = valid_class_id[b][:valid_num]
            outp[b, :valid_num, 1] = valid_score[b][:valid_num]
            outp[b, :valid_num, 2:6] = out_coords

        if self.quantized:
            score_scale = int(self.params['score_scale_value'])
            score_shift = int(self.params['score_shift_value'])
            outp[b, :valid_num, 1] = (outp[b, :valid_num, 1].int() +
                                      self.inputs[0].zerop).long() * score_scale >> score_shift
            outp[b, :valid_num, 1] = torch.clamp(outp[b, :valid_num, 1], self.outputs[0].qmin, self.outputs[0].qmax)

    self.outputs[0].betensor = outp
    self.outputs[1].betensor = valid_batch

    return (outp, valid_batch)


def get_lut(in_scale, in_zerop, out_scale, out_zerop, var, lut_in_dtype, lut_size_bits, clamp_min, clamp_max):
    lsteps = 2 ** lut_size_bits
    in_qmin, in_qmax = dtype2range(lut_in_dtype)
    lut = linear_dequantize(torch.linspace(in_qmin, in_qmax, steps=lsteps), in_scale, in_zerop)
    lut = lut / var
    lut = torch.exp(lut)
    lut = linear_quantize_clip(lut, out_scale, out_zerop, clamp_min, clamp_max).short()
    return lut


def calculate_box_quantization(self, rois, delta, STD_DIV):
    q_bits_activation = self.attrs["q_bits_activation"]
    q_mode_activation = self.attrs["q_mode_activation"]

    batch_inp_rois_scales,   batch_inp_rois_zp = rois.scale, rois.zerop
    batch_inp_deltas_scales, batch_inp_deltas_zp = delta.scale, delta.zerop

    delta_dtype = delta.dtype
    delta_qbits = delta.qbits
    rois_dtype = rois.dtype
    rois_qbits = rois.qbits

    scale_ty_out = batch_inp_deltas_scales * STD_DIV[0]
    scale_tx_out = batch_inp_deltas_scales * STD_DIV[1]
    delta_qmin, delta_qmax = bits2range(delta_qbits, True)
    # poposal delta variance >> detection delta, so we need limit exp(x), if x too large, will affect acc
    # and x is too large, it is meaningless, so limit max delta which is calculated according to scale_thtw_out
    # we hope scale_thtw_out is at least 10
    max_delta_th = torch.tensor(3276.0).log().item()*STD_DIV[2]
    max_delta_tw = torch.tensor(3276.0).log().item()*STD_DIV[3]
    delta_qmax_add_zp = delta_qmax + batch_inp_deltas_zp
    if (delta_qmax_add_zp > batch_inp_deltas_scales*max_delta_th) \
            or (delta_qmax_add_zp > batch_inp_deltas_scales*max_delta_tw):
        batch_inp_deltas_scales = delta_qmax_add_zp/max(max_delta_th, max_delta_tw)

    th_tensor = PyTensor(self.name + '/th_tensor')
    tw_tensor = PyTensor(self.name + '/tw_tensor')
    th_tensor.min = 0
    tw_tensor.min = 0
    if 'max_exp_thtw' in self.params:
        th_tensor.max = self.params['max_exp_thtw']
        tw_tensor.max = self.params['max_exp_thtw']
    else:
        th_tensor.max = torch.tensor(delta_qmax_add_zp/batch_inp_deltas_scales/STD_DIV[2]).exp()
        tw_tensor.max = torch.tensor(delta_qmax_add_zp/batch_inp_deltas_scales/STD_DIV[3]).exp()
    th_tensor.scale, th_tensor.zerop, th_tensor.qmin, th_tensor.qmax, th_tensor.dtype = \
        get_linear_quant_params_from_tensor(th_tensor, QuantMode.to_symmetric(q_mode_activation), 16, True)
    tw_tensor.scale, tw_tensor.zerop, tw_tensor.qmin, tw_tensor.qmax, tw_tensor.dtype = \
        get_linear_quant_params_from_tensor(tw_tensor, QuantMode.to_symmetric(q_mode_activation), 16, True)

    lut_size_bits = min(delta_qbits, int(self.get_attrs('lut_items_in_bits')))
    th_lut = get_lut(batch_inp_deltas_scales, batch_inp_deltas_zp, th_tensor.scale,
                     th_tensor.zerop, STD_DIV[2], delta_dtype, lut_size_bits, 0, 32767)
    tw_lut = get_lut(batch_inp_deltas_scales, batch_inp_deltas_zp, tw_tensor.scale,
                     tw_tensor.zerop, STD_DIV[3], delta_dtype, lut_size_bits, 0, 32767)
    self.constants["th_lut"] = PyTensor(self.name+"/th_lut", th_lut)
    self.constants["tw_lut"] = PyTensor(self.name+"/tw_lut", tw_lut)

    # scale for detla (x,y,w,h)
    y_scale, y_scale_type, y_shift, y_shift_type = \
        get_scale_approximation_params(1/scale_ty_out, mult_bits=16, force_shift_positive=self.force_shift_positive)
    x_scale, x_scale_type, x_shift, x_shift_type = \
        get_scale_approximation_params(1/scale_tx_out, mult_bits=16, force_shift_positive=self.force_shift_positive)
    h_scale, h_scale_tyep, h_shift, h_shift_type = \
        get_scale_approximation_params(1/th_tensor.scale, mult_bits=16, force_shift_positive=self.force_shift_positive)
    w_scale, w_scale_tyep, w_shift, w_shift_type = \
        get_scale_approximation_params(1/tw_tensor.scale, mult_bits=16, force_shift_positive=self.force_shift_positive)
    h_shift += 1  # for half calculate
    w_shift += 1
    # convert input box scale(batch_inp_rois_scales) to output box (32767)
    anchor_scale, anchor_scale_type, anchor_shift, anchor_shift_type = \
        get_scale_approximation_params(32767/batch_inp_rois_scales, mult_bits=14,
                                       force_shift_positive=self.force_shift_positive)

    self.params["box_scale_value"] = [int(y_scale), int(x_scale), int(h_scale), int(w_scale), int(anchor_scale)]
    self.params["box_scale_type"] = [y_scale_type, x_scale_type, h_scale_tyep, w_scale_tyep, anchor_scale_type]
    self.params["box_shift_value"] = [int(y_shift), int(x_shift), int(h_shift), int(w_shift), int(anchor_shift)]
    self.params["box_shift_type"] = [y_shift_type, x_shift_type, h_shift_type, w_shift_type, anchor_shift_type]

    delta_shift = rois_qbits+delta_qbits+17-32
    if batch_inp_deltas_zp != 0:
        delta_shift += 1
    self.params['delta_shift'] = delta_shift


@quant_register(OpType.MultiboxTransformLoc)
def multibox_transform_loc_quantize(self, *args):
    inp = self.inputs[:]
    out = self.outputs[:]

    std = [1/x for x in self.params['variance']]
    STD_DIV = [std[1], std[0], std[3], std[2]]  # [y,x,h,w]

    calculate_box_quantization(self, inp[2], inp[1], STD_DIV)

    score_threshold = self.params.pop('score_threshold')
    self.params['score_threshold_value'] = linear_quantize_clip(
        score_threshold, inp[0].scale, inp[0].zerop, inp[0].qmin, inp[0].qmax).int().item()
    self.params['score_threshold_type'] = inp[0].dtype

    score_scale, score_scale_type, score_shift, score_shift_type = \
        get_scale_approximation_params(32767/inp[0].scale, mult_bits=15,
                                       force_shift_positive=self.force_shift_positive)
    self.params['score_scale_value'] = int(score_scale)
    self.params['score_scale_type'] = score_scale_type
    self.params['score_shift_value'] = int(score_shift)
    self.params['score_shift_type'] = score_shift_type

    if 'std_div' in self.params:
        self.params.pop('std_div')

    # set dtpye and qbits
    out[0].scale, out[0].zerop, out[0].dtype, out[0].qbits = 32767, 0, Dtype.INT16, 16
    out[0].qinvariant = inp[0].qinvariant
    out[1].scale, out[1].zerop, out[1].dtype, out[1].qbits = 1, 0, Dtype.UINT16, 16
    out[1].qinvariant = True
