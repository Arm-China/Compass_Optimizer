# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
import numpy as np
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.utils.dtype_utils import construct_torch_tensor as torch_tensor
from AIPUBuilder.Optimizer.utils.quant_tool_utils import *
from AIPUBuilder.Optimizer.utils.math_utils import lookup_lut_powerof2
from AIPUBuilder.Optimizer.logger import *
import functools

'''layer_id=103
layer_name=SSD_DecodeBox
layer_type=DecodeBox
layer_bottom=decodebox_activation,concat_0
layer_bottom_shape=[1,1917,91],[1,1917,4]
layer_bottom_type=uint8,int8
layer_top=SSD_DecodeBox_box,SSD_DecodeBox_total_prebox,SSD_DecodeBox_total_class,SSD_DecodeBox_out_score,SSD_DecodeBox_label_perbox
layer_top_shape=[1,5000,4],[1,5000],[1,1],[1,5000],[1,5000]
layer_top_type=int16,uint16,uint16,uint8,uint16
weights_type=int8
weights_offset=16924508
weights_size=17384
weights_shape=8692
image_width=800
image_height=800
score_threshold_uint8=127
box_shift=13
proposal_normalized = true(default)
'''


_param_default_value = {
    'class_num': 21,
    'feature_map': [19, 10, 5, 3, 2, 1],
    'max_box_num': 5000,
    'variance': [10., 10., 5., 5.],
    'score_threshold': 0,
    'height': 300,
    'width': 300,
}


def decode_box(bbox, anchor, param, quantized):

    h_min = param['h_min']
    h_max = param['h_max']
    w_min = param['w_min']
    w_max = param['w_max']
    clip = param['clip']

    ty, tx, th, tw = bbox[:, :, 0], bbox[:, :, 1], bbox[:, :, 2], bbox[:, :, 3]
    if not quantized:
        variance = param['variance']
        ya, xa, ha, wa = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        ty = ty / variance[0]
        tx = tx / variance[1]
        th = th / variance[2]
        tw = tw / variance[3]

        h = torch.exp(th) * ha
        w = torch.exp(tw) * wa
        cy = ty * ha + ya
        cx = tx * wa + xa

        h = h.clamp(0, h_max - h_min)
        w = w.clamp(0, w_max - w_min)

        ymin = cy - h / 2.
        xmin = cx - w / 2.
        ymax = cy + h / 2.
        xmax = cx + w / 2.

    else:
        dev = bbox.device
        wa, ha, ya, xa = anchor[0, :], anchor[1, :], anchor[2, :], anchor[3, :]

        ty_lut, tx_lut, th_lut, tw_lut = param['ty_lut'], param['tx_lut'], param['th_lut'], param['tw_lut']
        shift = param['box_shift']

        lut_in_bits = param['lut_in_bits']
        lut_out_bits = param['lut_out_bits']
        in_is_signed = param['in_is_signed']
        out_is_signed = param['out_is_signed']

        lut_h = lookup_lut_powerof2(th, th_lut, lut_in_bits, in_is_signed, lut_out_bits, out_is_signed)
        lut_w = lookup_lut_powerof2(tw, tw_lut, lut_in_bits, in_is_signed, lut_out_bits, out_is_signed)
        lut_y = lookup_lut_powerof2(ty, ty_lut, lut_in_bits, in_is_signed, lut_out_bits, out_is_signed)
        lut_x = lookup_lut_powerof2(tx, tx_lut, lut_in_bits, in_is_signed, lut_out_bits, out_is_signed)

        h = (lut_h.to(dev).int() * ha.to(dev).int()) >> shift
        w = (lut_w.to(dev).int() * wa.to(dev).int()) >> shift
        cy = ((lut_y.to(dev).int() * ha.to(dev).int()) >> shift).int() + ya.to(dev).int()
        cx = ((lut_x.to(dev).int() * wa.to(dev).int()) >> shift).int() + xa.to(dev).int()

        h = h.clamp(0, h_max - h_min)
        w = w.clamp(0, w_max - w_min)

        ymin = cy - (h >> 1)
        xmin = cx - (w >> 1)
        ymax = cy + (h >> 1)
        xmax = cx + (w >> 1)
    if clip:
        ymin = ymin.clamp(h_min, h_max)
        xmin = xmin.clamp(w_min, w_max)
        ymax = ymax.clamp(h_min, h_max)
        xmax = xmax.clamp(w_min, w_max)

    boxes = torch.stack([ymin, xmin, ymax, xmax], axis=-1)
    stats_coords, stats_boxes = None, None
    if not quantized:
        stats_coords = torch.cat([torch.unsqueeze(ya, dim=0),
                                  torch.unsqueeze(xa, dim=0),
                                  torch.unsqueeze(ha, dim=0),
                                  torch.unsqueeze(wa, dim=0),
                                  ymin, xmin, ymax, xmax, cy, cx, h, w, ty*ha, tx*wa], dim=0)
        stats_boxes = torch.stack([ty, tx, th, tw, torch.exp(th), torch.exp(tw)])

    return boxes, stats_coords, stats_boxes


float_decode_box = functools.partial(decode_box, quantized=False)
quant_decode_box = functools.partial(decode_box, quantized=True)


def get_roi_one_batch(dec_boxes, class_score, roi_boxes, roi_scores, score_threshold, max_box_num, max_class_num, real_class_num):

    # exclude the backgroud if have
    bg = class_score.shape[1] - real_class_num
    cls_score = class_score[:, bg:]
    if score_threshold is None or score_threshold == 0:
        # TODO return top1
        OPT_WARN('Donot support score_threshold = None or =0 temporally in decodebox_ssd')
        return None
    else:
        prop_index = torch.where(cls_score[:, 0:] > score_threshold)
        box_ids, class_ids = prop_index[0], prop_index[1]

        prop_num = box_ids.shape[0]
        if prop_num > max_box_num:
            prop_score = cls_score[:, 0:][prop_index]
            score_sort_idx = torch.argsort(prop_score, dim=-1)
            first_max_num_score = score_sort_idx[:max_box_num]
            box_ids = box_ids[first_max_num_score]
            class_ids = class_ids[first_max_num_score]

        label_perclass, box_num_perclass = torch.unique(class_ids, return_counts=True)
        total_class_num = label_perclass.numel()
        start_ps = torch.cumsum(box_num_perclass, dim=0) - box_num_perclass
        end_ps = torch.cumsum(box_num_perclass, dim=0)
        for t in range(total_class_num):
            idx = torch.where(class_ids == label_perclass[t])
            cls_idx = class_ids[idx]
            box_idx = box_ids[idx]
            roi_boxes[start_ps[t]:end_ps[t], :] = dec_boxes[box_idx, :]
            roi_scores[start_ps[t]:end_ps[t]] = cls_score[box_idx, cls_idx]

        label_perclass_all = torch.zeros(max_class_num)
        box_num_perclass_all = torch.zeros(max_class_num)
        label_perclass_all[:label_perclass.shape[0]] = label_perclass
        box_num_perclass_all[:box_num_perclass.shape[0]] = box_num_perclass

        return roi_boxes, roi_scores, box_num_perclass_all, label_perclass_all, torch_tensor(total_class_num, device=roi_boxes.device)


def get_roi(boxes, scores, threshold, max_box_num, max_class_num, real_class_num):
    batch_size = scores.shape[0]
    roi_boxes = torch.zeros([batch_size, max_box_num, boxes.shape[2]], dtype=boxes.dtype, device=boxes.device)
    roi_scores = torch.zeros([batch_size, max_box_num], dtype=scores.dtype, device=boxes.device)
    box_num_perclass = torch.zeros([batch_size, max_class_num], dtype=torch.float32, device=boxes.device)
    label_perclass = torch.zeros([batch_size, max_class_num], dtype=torch.float32, device=boxes.device)
    total_class_num = torch.zeros([batch_size, 1], dtype=torch.float32, device=boxes.device)

    for bs in range(batch_size):
        ret = get_roi_one_batch(boxes[bs],
                                scores[bs],
                                roi_boxes[bs],
                                roi_scores[bs],
                                threshold,
                                max_box_num,
                                max_class_num,
                                real_class_num)
        roi_boxes[bs], roi_scores[bs], box_num_perclass[bs], label_perclass[bs], total_class_num[bs] = ret[0],\
            ret[1],\
            ret[2],\
            ret[3],\
            ret[4]
    return [roi_boxes, box_num_perclass, total_class_num, roi_scores, label_perclass]


@op_register(OpType.DecodeBox)
def decodebox_ssd(self, *args):

    class_score = self.inputs[0].betensor
    bbox = self.inputs[1].betensor
    dev = class_score.device
    config = {}

    if len(self.inputs) == 3:
        anchor_box = self.inputs[2].betensor
    else:
        ha = self.constants['ha'].betensor
        wa = self.constants['wa'].betensor
        xcenter = self.constants['xcenter'].betensor
        ycenter = self.constants['ycenter'].betensor

    if not self.quantized:
        score_threshold = self.get_param('score_threshold')
        variance = self.get_param('variance', optional=True, default_value=_param_default_value['variance'])
        proposal_normalized = self.get_param('proposal_normalized', optional=True, default_value=True)
        config['h_min'], config['w_min'] = 0., 0.
        config['h_max'] = 1.0 if proposal_normalized else self.get_param('image_height')
        config['w_max'] = 1.0 if proposal_normalized else self.get_param('image_width')
        config['clip'] = self.get_param('clip', optional=True, default_value=True)
        config['variance'] = variance
        if len(self.inputs) < 3:
            anchor_box = torch.stack([ycenter, xcenter, ha, wa], dim=1)
        decode_boxes, stats_coords, stats_boxes = float_decode_box(bbox, anchor_box, config)

        if len(self.placeholders) == 0:
            sc_t = PyTensor(self.name+'_stat_coords', stats_coords, dtype=Dtype.FP32)
            sb_t = PyTensor(self.name+'_stat_boxes', stats_boxes, dtype=Dtype.FP32)
            sc_t.dtype = Dtype.FP32
            sb_t.dtype = Dtype.FP32
            self.placeholders.append(sc_t)
            self.placeholders.append(sb_t)
        else:
            self.placeholders[0].betensor = stats_coords
            self.placeholders[1].betensor = stats_boxes
    else:
        class_score = self.inputs[0].betensor + self.inputs[0].zerop
        config['ty_lut'] = self.constants['ty_lut'].betensor
        config['tx_lut'] = self.constants['tx_lut'].betensor
        config['th_lut'] = self.constants['th_lut'].betensor
        config['tw_lut'] = self.constants['tw_lut'].betensor
        config['lut_in_bits'] = self.inputs[1].qbits
        config['lut_out_bits'] = dtype2bits(self.get_constant('ty_lut').dtype)
        config['in_is_signed'] = is_signed(self.inputs[1].dtype)
        config['out_is_signed'] = is_signed(self.inputs[1].dtype)

        config['h_min'] = 0
        config['h_max'] = self.get_param('image_height')
        config['w_min'] = 0
        config['w_max'] = self.get_param('image_width')
        config['clip'] = self.get_param('clip', optional=True, default_value=True)
        config['box_shift'] = self.get_param('box_shift')

        if len(self.inputs) < 3:
            anchor_box = torch.stack([wa, ha, ycenter, xcenter], dim=0)
        decode_boxes, _, _ = quant_decode_box(bbox, anchor_box, config)
        score_threshold = self.params['score_threshold_uint8']

    # params['max_box_num'] #This parameter is not used at present because it takes time and the nms will do it later
    max_box_num = self.outputs[0].ir_shape[1]
    max_class_num = self.outputs[1].ir_shape[1]
    class_num = self.get_param('class_num')
    outputs = get_roi(decode_boxes, class_score, score_threshold, max_box_num, max_class_num, class_num)
    # boxes, box_num_perclass, total_class_num, roi_scores, label_perclass
    zerops = [torch_tensor(0., device=self.outputs[0].device)] * 5
    if self.quantized:
        for zp, out in zip(zerops, self.outputs):
            zp = out.broadcast_zerop
    self.outputs[0].betensor = outputs[0] - zerops[0]
    self.outputs[1].betensor = outputs[1] - zerops[1]
    self.outputs[2].betensor = outputs[2] - zerops[2]
    self.outputs[3].betensor = outputs[3] - zerops[3]
    self.outputs[4].betensor = outputs[4] - zerops[4]

    return [o.betensor for o in self.outputs]


def get_bbox_lut(in_scale, in_zerop, var, box_scale, lut_in_dtype, lut_size_bits, lut_range_dtype, flag):
    lsteps = 2 ** lut_size_bits
    in_qmin, in_qmax = dtype2range(lut_in_dtype)
    lut_o_qmin, lut_o_qmax = dtype2range(lut_range_dtype)
    lut = linear_dequantize(torch.linspace(in_qmin, in_qmax, steps=lsteps), in_scale, in_zerop)
    lut = lut / var
    if flag:
        lut = torch.exp(lut) * box_scale
    else:
        lut = lut * box_scale
    lut = torch.clamp(torch.round(lut), lut_o_qmin, lut_o_qmax)  # .short()
    return lut


@quant_register(OpType.DecodeBox)
def quantize_decodebox(self, *args):
    """
        step1: quantize score_threshold
        step2: if prior anchor in attrs['weights'], using statistic coords to get enlarge_scale,
               then qanchor=fp32_anchor * enlarge_scale
        step3: quantize exp, using statistic box to get enlarge_scale, then get int16 lut.
        step4: concat(qanchor, lut) to constant
        step5: update output Tensor's attrs
    """
    # prepare params
    in_score = self.inputs[0]
    in_bbox = self.inputs[1]
    q_mode_out = self.attrs['q_mode_activation']

    extra_params = self.get_attrs('extra_params', optional=True, default_value=[0, 16, 16, 16, 16])
    if QuantMode.is_per_channel(q_mode_out) == True:
        OPT_ERROR(self.type + " currently Not support per-channel quantization")

    # box, box_num_perClass, total_class_num, score, label_perClass
    # _default_qbits_list = [16, 16, 16, in_score.qbits, 16]
    _default_qbits_list = extra_params[1:-1] + [in_score.qbits] + extra_params[-1:]
    q_bits_activation = self.attrs['q_bits_activation']
    _qbits_list = [max(default_bits, q_bits_activation) for default_bits in _default_qbits_list]
    _qbits_list[3] = in_score.qbits

    # step1
    score_threshold = self.get_param('score_threshold')  # int(score_threshold * (2 ** in_score.qbits - 1.0))
    self.params['score_threshold_uint8'] = linear_quantize_clip(
        score_threshold, in_score.scale, in_score.zerop, in_score.qmin, in_score.qmax).int().item()

    # step2
    q_in_anchor = None
    stats_coords = self.placeholders[0]
    stats_boxes = self.placeholders[1]
    stats_coords.dtype = Dtype.FP32
    stats_boxes.dtype = Dtype.FP32
    stats_coords.qinvariant = False
    stats_boxes.qinvariant = False
    if len(self.inputs) == 3:
        in_anchor = self.inputs[2]
        coord_scale = in_anchor.scale
        coord_zp = in_anchor.zerop
    else:
        anchor_name = ['ha', 'wa', 'xcenter', 'ycenter']
        ha = self.constants['ha'].betensor
        wa = self.constants['wa'].betensor
        xcenter = self.constants['xcenter'].betensor
        ycenter = self.constants['ycenter'].betensor
        anchor_box = torch.stack([ycenter, xcenter, ha, wa], dim=1)
        anchor_Pytensor = PyTensor(self.name+"/anchor_box", anchor_box, dtype=Dtype.FP32)
        anchor_Pytensor.max = anchor_box.max().item()
        anchor_Pytensor.min = anchor_box.min().item()
        sign = is_signed(in_bbox.dtype)
        stats_coords.qbits = _qbits_list[0]
        qstats = get_linear_quant_params_from_tensor(
            stats_coords, 'per_tensor_symmetric_restricted_range', _qbits_list[0], sign)
        stats_coords.scale, stats_coords.zerop, stats_coords.qmin, stats_coords.qmax, stats_coords.dtype = qstats

        coord_scale = max(1, 2**(torch.floor(torch.log2(stats_coords.scale))).int().item())
        for name in anchor_name:
            self.constants[name].scale, self.constants[name].zerop, self.constants[name].qmin, self.constants[name].qmax, self.constants[name].dtype = qstats
            self.constants[name].qbits = stats_coords.qbits
            self.constants[name].qinvariant = in_bbox.qinvariant
            self.constants[name].betensor = torch.clamp(
                self.constants[name].betensor * coord_scale, stats_coords.qmin, stats_coords.qmax).short()

        coord_zp = stats_coords.zerop

    proposal_normalized = self.get_param('proposal_normalized', optional=True, default_value=True)
    height_max = 1 if proposal_normalized else self.get_param('image_height')
    width_max = 1 if proposal_normalized else self.get_param('image_width')
    self.params['image_height'] = coord_scale * height_max
    self.params['image_width'] = coord_scale * width_max

    # step3
    sign = is_signed(in_bbox.dtype)
    stats_boxes.qbits = _qbits_list[0]
    qstats = get_linear_quant_params_from_tensor(
        stats_boxes, 'per_tensor_symmetric_restricted_range', stats_boxes.qbits, sign)
    stats_boxes.scale, stats_boxes.zerop, stats_boxes.qmin, stats_boxes.qmax = qstats[0],\
        qstats[1],\
        qstats[2],\
        qstats[3]

    box_shift = torch.floor(torch.log2(stats_boxes.scale)).int().item()
    box_scale = 2 ** box_shift

    in_scale = in_bbox.scale
    in_zerop = in_bbox.zerop
    var = self.get_param('variance', optional=True, default_value=_param_default_value['variance'])
    lut_in_dtype = in_bbox.dtype
    lut_size_bits = min(in_bbox.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut_range_bits = max(self.attrs['q_bits_activation'], 16)
    lut_out_dtype = bits2dtype(lut_range_bits, True)
    lut_out_udtype = bits2dtype(lut_range_bits, False)

    ty_lut = get_bbox_lut(in_scale, in_zerop, var[0], box_scale, lut_in_dtype, lut_size_bits, lut_out_dtype, False)
    tx_lut = get_bbox_lut(in_scale, in_zerop, var[1], box_scale, lut_in_dtype, lut_size_bits, lut_out_dtype, False)
    th_lut = get_bbox_lut(in_scale, in_zerop, var[2], box_scale, lut_in_dtype, lut_size_bits, lut_out_udtype, True)
    tw_lut = get_bbox_lut(in_scale, in_zerop, var[3], box_scale, lut_in_dtype, lut_size_bits, lut_out_udtype, True)

    lut_name = ['ty_lut', 'tx_lut', 'th_lut', 'tw_lut']
    for name in lut_name:
        self.constants[name] = PyTensor(self.name+name, eval(name), dtype=lut_out_dtype)

    self.params['box_shift'] = box_shift

    # step5
    _oscale_list = [coord_scale, 1.0, 1.0, in_score.scale, 1.0]
    _ozerop_list = [coord_zp, 0., 0., in_score.zerop, 0.0]
    _osigned_list = [True, False or self.force_dtype_int,
                     False or self.force_dtype_int, is_signed(in_score.dtype), True]
    _dtype_list = [bits2dtype(b, s) for b, s in zip(_qbits_list, _osigned_list)]

    for i, o in enumerate(self.outputs):
        o.scale = _oscale_list[i]
        o.zerop = _ozerop_list[i]
        o.dtype = _dtype_list[i]
        o.qbits = _qbits_list[i]
    self.outputs[0].qinvariant = False
    self.outputs[1].qinvariant = True
    self.outputs[2].qinvariant = True
    self.outputs[3].qinvariant = in_score.qinvariant
    self.outputs[4].qinvariant = True
