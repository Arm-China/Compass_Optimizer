# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch

'''
layer_type=DetectionOutput
layer_bottom=[cls_prob_post_reshape,bbox_pred_post_reshape_post_reshape_post_slice,proposal_reshape_post_reshape_post_slice]
layer_bottom_shape=[[1,300,81],[1,300,80,4],[1,300,4]]
layer_bottom_type=[float32,float32,float32]
layer_top=[fasterrcnn_new_3_detection_0,fasterrcnn_new_3_detection_1,fasterrcnn_new_3_detection_2,fasterrcnn_new_3_detection_3,fasterrcnn_new_3_detection_4]
layer_top_shape=[[1,5000],[1,5000,4],[1,5000],[1,5000],[1,1]]
layer_top_type=[float32,float32,float32,float32,float32]
image_width=600
image_height=600
score_threshold=0.800000
'''


def FAIL_LOG(result, log):
    if result:
        OPT_FATAL(log)


def get_bottom_nodes(node):
    input_num = len(node.inputs)
    if input_num != 3:
        OPT_FATAL("input count must be 3")

    if node.quantized:
        class_conf = node.inputs[0].betensor.int()
        box_encodings = node.inputs[1].betensor.int()
        crop_proposals = node.inputs[2].betensor.int()
    else:
        class_conf = node.inputs[0].betensor.float()
        box_encodings = node.inputs[1].betensor.float()
        crop_proposals = node.inputs[2].betensor.float()

    FAIL_LOG(class_conf.dim() != 3, 'the class confidence must have 3-dimention!')
    FAIL_LOG(box_encodings.dim() != 4, 'the box encodings must have 4-dimention!')
    FAIL_LOG(crop_proposals.dim() != 3, 'the crop proposals must have 3-dimention!')

    return class_conf, box_encodings, crop_proposals


def get_proposals_box(crop_proposals, quantized=False):
    coord1 = (crop_proposals[:, :, 0] + crop_proposals[:, :, 2])
    coord2 = (crop_proposals[:, :, 1] + crop_proposals[:, :, 3])
    len1 = crop_proposals[:, :, 2] - crop_proposals[:, :, 0]
    len2 = crop_proposals[:, :, 3] - crop_proposals[:, :, 1]  # [300]

    if quantized:
        coord1 = coord1 >> 1
        coord2 = coord2 >> 1
    else:
        coord1 = coord1 / 2.
        coord2 = coord2 / 2.

    return torch.unsqueeze(coord1, 1), torch.unsqueeze(coord2, 1),\
        torch.unsqueeze(len1, 1), torch.unsqueeze(len2, 1)


def get_encodings_box(box_encodings, box_encoding_coef):
    coord1 = box_encodings[:, :, :, 0] / box_encoding_coef[0]
    coord2 = box_encodings[:, :, :, 1] / box_encoding_coef[1]
    len1 = box_encodings[:, :, :, 2] / box_encoding_coef[2]
    len2 = box_encodings[:, :, :, 3] / box_encoding_coef[3]
    return coord1, coord2, len1, len2


def caffe_detection(node, class_conf, box_encodings, crop_proposals):
    res = []
    box_encoding_coef = node.get_param('variance')  # [1.0, 1.0, 1.0, 1.0]

    tx, ty, tw, th = get_encodings_box(box_encodings, box_encoding_coef)
    res.extend([ty, tx, tw, th])

    if not node.quantized:
        xcenter_a, ycenter_a, wa, ha = get_proposals_box(crop_proposals)
        res.extend([xcenter_a, ycenter_a, wa, ha])

    return res


def default_detect(node, class_conf, box_encodings, crop_proposals):
    res = []
    if node.quantized:
        box_encoding_coef = [1, 1, 1, 1]
        ty, tx, th, tw = get_encodings_box(box_encodings, box_encoding_coef)
        res.extend([ty, tx, tw, th])
    else:
        box_encoding_coef = node.get_param('variance')
        ty, tx, th, tw = get_encodings_box(box_encodings, box_encoding_coef)
        res.extend([ty, tx, tw, th])
        ycenter_a, xcenter_a, ha, wa = get_proposals_box(crop_proposals)
        res.extend([xcenter_a, ycenter_a, wa, ha])

    return res


def _get_box_score(class_score, box_encoding, score_thresh, max_detection_num, max_class_num):
    class_num = class_score.shape[-1]
    crop_box_num = class_score.shape[0]
    box = torch.zeros((max_detection_num, 4), dtype=torch.float32, device=class_score.device)
    score = torch.zeros((max_detection_num), dtype=torch.float32, device=class_score.device)
    box_num_perClass = torch.zeros([max_class_num], dtype=torch.float32, device=class_score.device)

    outbox_idx = 0
    box_num_perClass_list = []
    class_label = []
    for class_idx in range(class_num):
        # if class_idx == 0:
        #     continue
        box_num_curClass = 0
        for box_idx in range(crop_box_num):
            if class_score[box_idx][class_idx] > score_thresh:
                if outbox_idx >= max_detection_num:
                    break
                if class_idx not in class_label:
                    class_label.append(class_idx)
                box[outbox_idx, :] = box_encoding[box_idx * class_num + class_idx, :]
                score[outbox_idx] = class_score[box_idx, class_idx]
                outbox_idx += 1
                box_num_curClass += 1
        if box_num_curClass != 0:
            box_num_perClass_list.append(box_num_curClass)
    box_num_perClass[:len(box_num_perClass_list)] = torch.tensor(box_num_perClass_list, device=class_score.device)
    total_class_num = len(class_label)
    class_label.extend((max_class_num - total_class_num) * [0])
    return box, score, box_num_perClass, \
        torch.tensor(class_label, device=class_score.device), \
        torch.tensor(total_class_num, device=class_score.device)


def get_box_score(batch_class_score, batch_box_encoding, score_thresh, max_box_num, max_class_num):
    batch_size = batch_class_score.shape[0]
    max_detection_num = max_box_num
    box = torch.zeros((batch_size, max_detection_num, batch_box_encoding.shape[2]),
                      device=batch_class_score.device)
    score = torch.zeros((batch_size, max_detection_num), device=batch_class_score.device)
    box_num_perClass = torch.zeros((batch_size, max_class_num), device=batch_class_score.device)
    class_label = torch.zeros([batch_size, max_class_num], dtype=torch.int, device=batch_class_score.device)
    total_class_num = torch.ones([batch_size, 1], dtype=torch.int, device=batch_class_score.device)

    for i in range(batch_class_score.shape[0]):
        box[i], score[i], box_num_perClass[i], class_label[i], total_class_num[i] = _get_box_score(batch_class_score[i],
                                                                                                   batch_box_encoding[i],
                                                                                                   score_thresh,
                                                                                                   max_box_num,
                                                                                                   max_class_num)
    return box, score, box_num_perClass, class_label, total_class_num


@op_register(OpType.DetectionOutput)
def detectionoutput(self, *args):
    inp = self.inputs[0].betensor

    class_conf, box_encodings, crop_proposals = get_bottom_nodes(self)
    if class_conf.shape[2] > 1:
        class_conf = class_conf[:, :, 1:]
    if class_conf.shape[2] != box_encodings.shape[2]:
        box_encodings = box_encodings[:, :, 1:, :]
    batch_size = box_encodings.shape[0]

    special_anchor_mode = self.get_param('anchor_mode') if 'anchor_mode' in self.params else 'default_detect'
    if special_anchor_mode == 'caffe_detection':
        res_box = caffe_detection(self, class_conf, box_encodings, crop_proposals)
    else:
        res_box = default_detect(self, class_conf, box_encodings, crop_proposals)

    ty = res_box[0].permute(0, 2, 1)
    tx = res_box[1].permute(0, 2, 1)
    tw = res_box[2].permute(0, 2, 1)
    th = res_box[3].permute(0, 2, 1)

    if not self.quantized:
        height = self.get_param('image_height')
        width = self.get_param('image_width')
        score_thresh = self.get_param('score_threshold')
        bbox_xform_clip = self.get_param('bbox_xform_clip', optional=True, default_value=float('inf'))
        th = torch.clamp(th, max=bbox_xform_clip)
        tw = torch.clamp(tw, max=bbox_xform_clip)
        crop_box_num = class_conf.shape[1]
        w = torch.exp(tw) * res_box[6]
        h = torch.exp(th) * res_box[7]
        ycenter = ty * res_box[7] + res_box[5]
        xcenter = tx * res_box[6] + res_box[4]

        # upper left:[ymin,xmin], lower right:[ymax,xmin]
        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0

        ymin = torch.clamp(ymin, 0, height)
        ymax = torch.clamp(ymax, 0, height)
        xmin = torch.clamp(xmin, 0, width)
        xmax = torch.clamp(xmax, 0, width)

        coords_stats = [ymin, ymax, xmin, xmax, ycenter, xcenter, w, h]
        box_stats = [torch.exp(th), torch.exp(tw)]
        txty_stats = [ty, tx]
        placeholders = [coords_stats, box_stats, txty_stats]
        placeholders_output = []

        for placeholder in placeholders:
            tensor_all = placeholder[0]
            for idx, tensor in enumerate(placeholder):
                tensor = torch.reshape(tensor, (-1,))
                tensor_all = tensor if idx == 0 else torch.cat((tensor_all, tensor), dim=0)
            placeholders_output.append(tensor_all)

        if len(self.placeholders) < 1:
            ph0 = PyTensor(self.name+"/coords", placeholders_output[0].cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            ph1 = PyTensor(self.name+"/box", placeholders_output[1].cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            ph2 = PyTensor(self.name + "/txty", placeholders_output[2].cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            self.placeholders.append(ph0)
            self.placeholders.append(ph1)
            self.placeholders.append(ph2)
        self.placeholders[0].betensor = placeholders_output[0]
        self.placeholders[1].betensor = placeholders_output[1]
        self.placeholders[2].betensor = placeholders_output[2]
    else:
        dev = box_encodings.device
        tw_lut = self.constants['tw_lut'].betensor
        th_lut = self.constants['th_lut'].betensor
        ty_lut = self.constants['ty_lut'].betensor
        tx_lut = self.constants['tx_lut'].betensor

        height = self.get_param('height')
        width = self.get_param('width')
        score_thresh = self.params["score_threshold_value"]
        if special_anchor_mode == 'caffe_detection':
            xcenter_a_q, ycenter_a_q, wa_q, ha_q = get_proposals_box(crop_proposals, self.quantized)
        else:
            ycenter_a_q, xcenter_a_q, ha_q, wa_q = get_proposals_box(crop_proposals, self.quantized)

        lut_in_bits = self.inputs[1].qbits
        in_is_signed = is_signed(self.inputs[1].dtype)
        hlut_out_bits = dtype2bits(self.get_constant('th_lut').dtype)
        xlut_out_bits = dtype2bits(self.get_constant('tx_lut').dtype)
        hout_is_signed = is_signed(self.get_constant('th_lut').dtype)
        xout_is_signed = is_signed(self.get_constant('tx_lut').dtype)

        lut_h = lookup_lut_powerof2(th, th_lut, lut_in_bits, in_is_signed, hlut_out_bits, hout_is_signed)
        lut_w = lookup_lut_powerof2(tw, tw_lut, lut_in_bits, in_is_signed, hlut_out_bits, hout_is_signed)
        lut_y = lookup_lut_powerof2(ty, ty_lut, lut_in_bits, in_is_signed, xlut_out_bits, xout_is_signed)
        lut_x = lookup_lut_powerof2(tx, tx_lut, lut_in_bits, in_is_signed, xlut_out_bits, xout_is_signed)

        do_shift1, do_shift2, do_shift3 = self.params["box_shift_value"]
        do_scale1, do_scale2, do_scale3 = self.params["box_scale_value"]
        pre_shift1, pre_shift2 = self.get_param('delta_shift')

        round_shift1 = 1 << (do_shift1 - pre_shift1 - 1)
        round_shift2 = 1 << (do_shift2 - pre_shift2 - 1)
        h = torch.clamp((lut_h.to(dev).int() * do_scale1) >> pre_shift1, -2 ** 15, 2 ** 15 - 1).int()
        w = torch.clamp((lut_w.to(dev).int() * do_scale1) >> pre_shift1, -2 ** 15, 2 ** 15 - 1).int()
        h = ((h * ha_q) + round_shift1) >> (do_shift1 - pre_shift1)
        w = ((w * wa_q) + round_shift1) >> (do_shift1 - pre_shift1)
        cy = torch.clamp((lut_y.to(dev).int() * do_scale2) >> pre_shift2, -2 ** 15, 2 ** 15 - 1).int()
        cy = (((cy * ha_q) + round_shift2) >> (do_shift2 - pre_shift2)) + \
            linear_requantize(ycenter_a_q, do_scale3, do_shift3, 0, -2 ** 15, 2 ** 15 - 1)
        cx = torch.clamp((lut_x.to(dev).int() * do_scale2) >> pre_shift2, -2 ** 15, 2 ** 15 - 1).int()
        cx = (((cx * wa_q) + round_shift2) >> (do_shift2 - pre_shift2)) + \
            linear_requantize(xcenter_a_q, do_scale3, do_shift3, 0, -2 ** 15, 2 ** 15 - 1)
        ymin = cy - (h >> 1)
        xmin = cx - (w >> 1)
        ymax = cy + (h >> 1)
        xmax = cx + (w >> 1)

        ymin = torch.clamp(ymin, 0, height)
        ymax = torch.clamp(ymax, 0, height)
        xmin = torch.clamp(xmin, 0, width)
        xmax = torch.clamp(xmax, 0, width)

    proposal_box = torch.stack((ymin, xmin, ymax, xmax)).permute(
        1, 3, 2, 0).reshape([batch_size, xmax.shape[1] * xmax.shape[2], 4])

    detection_boxes, detection_scores, box_num_perClass, class_label, total_class_num = get_box_score(class_conf,
                                                                                                      proposal_box,
                                                                                                      score_thresh,
                                                                                                      max_box_num=self.outputs[0].ir_shape[1],
                                                                                                      max_class_num=self.outputs[2].ir_shape[1])
    out = self.outputs[:]
    out[0].betensor = (detection_scores)
    out[1].betensor = (detection_boxes)
    out[2].betensor = (box_num_perClass)
    out[3].betensor = (class_label)
    out[4].betensor = (total_class_num)
    return detection_scores, detection_boxes, box_num_perClass, class_label, total_class_num


def get_proposal_coords_lut(node, box_encoding_scale, scale_anchor, coord_scale, precision_bits, heightWidth=True):
    precision_q_min, precision_q_max = bits2range(precision_bits, is_signed=True)
    input_range = torch.arange(precision_q_min, precision_q_max + 1,
                               device=node.inputs[0].betensor.device) / box_encoding_scale
    xs = linear_dequantize(input_range, scale_anchor, 0)
    if heightWidth:
        # proposal_coords_lut = torch.exp(xs) * coord_scale
        xs = torch.exp(xs)
    proposal_coords_lut = linear_quantize_clip(xs, coord_scale, 0,
                                               -2 ** 15, 2 ** 15 - 1).type(torch.int16)
    return proposal_coords_lut


@quant_register(OpType.DetectionOutput)
def detectionoutput_quantize(self, *args):
    q_mode_bias = self.attrs["q_mode_bias"]
    q_mode_weight = self.attrs["q_mode_weight"]
    q_mode_activation = self.attrs["q_mode_activation"]
    if q_mode_weight != q_mode_bias:
        OPT_FATAL("Currently quantization mode of weight (q_mode_weight) and bias (q_mode_bias) must be the same!")
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_activation = self.attrs["q_bits_activation"]

    special_anchor_mode = self.get_param('anchor_mode') if 'anchor_mode' in self.params else 'default_detect'

    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    inp2 = self.inputs[2]

    height = self.get_param('image_height')
    width = self.get_param('image_width')
    score_thresh = self.get_param('score_threshold')
    self.params.pop('image_height')
    self.params.pop('image_width')
    self.params.pop('score_threshold')

    class_score_scale, class_score_zp = inp0.scale, inp0.zerop
    box_encoding_scale, box_encoding_zp = inp1.scale, inp1.zerop
    proposal_box_scale, proposal_box_zp = inp2.scale, inp2.zerop

    coords = self.placeholders[0]
    box = self.placeholders[1]
    tytx = self.placeholders[2]
    coords.qbits = max(16, q_bits_activation)
    coords.scale, coords.zerop, coords.qmin, coords.qmax, coords.dtype = get_linear_quant_params_from_tensor(
        coords, QuantMode.to_symmetric(q_mode_activation), coords.qbits, is_signed=True)
    coords.qinvariant = False
    stat_coords_scale, stat_coords_zp = coords.scale, coords.zerop
    # stat_coords_scale = 2 ** torch.floor(torch.log2(torch.tensor(stat_coords_scale))).item()

    self.params['height'] = int(height * stat_coords_scale)
    self.params['width'] = int(width * stat_coords_scale)

    # compute box scale
    box.qbits = max(16, q_bits_activation)
    box.scale, box.zerop, box.qmin, box.qmax, box.dtype = get_linear_quant_params_from_tensor(
        box, QuantMode.to_symmetric(q_mode_activation), box.qbits, is_signed=True)
    box.qinvariant = False
    coord_zerop = box.zerop
    coord_scale = box.scale
    coord_shift = torch.floor(torch.log2(torch.tensor(coord_scale))).item()
    # coord_scale = 2 ** coord_shift

    tytx.qbits = max(16, q_bits_activation)
    tytx.scale, tytx.zerop, tytx.qmin, tytx.qmax, tytx.dtype = get_linear_quant_params_from_tensor(
        tytx, QuantMode.to_symmetric(q_mode_activation), tytx.qbits, is_signed=True)
    tytx.qinvariant = False

    do_scale1, do_scale_type1, do_shift1, do_shift_type1 = get_scale_approximation_params(stat_coords_scale/(coord_scale * inp2.scale),
                                                                                          mult_bits=16,
                                                                                          force_shift_positive=self.force_shift_positive)
    do_scale2, do_scale_type2, do_shift2, do_shift_type2 = get_scale_approximation_params(stat_coords_scale/(tytx.scale * inp2.scale),
                                                                                          mult_bits=16,
                                                                                          force_shift_positive=self.force_shift_positive)
    do_scale3, do_scale_type3, do_shift3, do_shift_type3 = get_scale_approximation_params(stat_coords_scale/(inp2.scale),
                                                                                          mult_bits=16,
                                                                                          force_shift_positive=self.force_shift_positive)

    self.params["box_shift_type"] = [do_shift_type1, do_shift_type2, do_shift_type3]
    self.params["box_shift_value"] = [int(do_shift1), int(do_shift2), int(do_shift3)]
    self.params["box_scale_value"] = [int(do_scale1), int(do_scale2), int(do_scale3)]
    self.params["box_scale_type"] = [do_scale_type1, do_scale_type2, do_scale_type3]

    import math
    scale1_bits = int(math.ceil(math.log2(do_scale1)))
    scale2_bits = int(math.ceil(math.log2(do_scale2)))
    pre_shift1 = scale1_bits
    pre_shift2 = scale2_bits
    self.params["delta_shift"] = [pre_shift1, pre_shift2]

    def get_lut(in_scale, in_zerop, var, box_scale, box_zerop, lut_in_dtype, lut_size_bits, lut_range_dtype, flag, bbox_xform_clip):
        lsteps = 2 ** lut_size_bits
        in_qmin, in_qmax = dtype2range(lut_in_dtype)
        lut_o_qmin, lut_o_qmax = dtype2range(lut_range_dtype)
        lut = linear_dequantize(torch.linspace(in_qmin, in_qmax, steps=lsteps), in_scale, in_zerop)

        lut = lut / var
        if flag:
            lut = torch.clamp(lut, max=bbox_xform_clip)
            lut = torch.exp(lut)
        lut = linear_quantize_clip(lut, box_scale, box_zerop, lut_o_qmin, lut_o_qmax)
        return lut

    var_list = self.params['variance']
    lut_in_dtype = inp1.dtype
    lut_size_bits = min(inp1.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut_range_bits = max(q_bits_activation, 16)
    lut_out_dtype = bits2dtype(lut_range_bits, True)

    bbox_xform_clip = self.get_param('bbox_xform_clip', optional=True, default_value=float('inf'))
    ty_lut = get_lut(box_encoding_scale, box_encoding_zp,
                     var_list[0], tytx.scale, tytx.zerop, lut_in_dtype, lut_size_bits, lut_out_dtype, False, bbox_xform_clip)
    tx_lut = get_lut(box_encoding_scale, box_encoding_zp,
                     var_list[1], tytx.scale, tytx.zerop, lut_in_dtype, lut_size_bits, lut_out_dtype, False, bbox_xform_clip)
    th_lut = get_lut(box_encoding_scale, box_encoding_zp,
                     var_list[2], coord_scale, coord_zerop, lut_in_dtype, lut_size_bits, lut_out_dtype, True, bbox_xform_clip)
    tw_lut = get_lut(box_encoding_scale, box_encoding_zp,
                     var_list[3], coord_scale, coord_zerop, lut_in_dtype, lut_size_bits, lut_out_dtype, True, bbox_xform_clip)
    if 'bbox_xform_clip' in self.params:
        self.params.pop('bbox_xform_clip')

    lut_list = [ty_lut, tx_lut, th_lut, tw_lut]
    lut_object_name = {ty_lut: 'ty_lut',
                       tx_lut: 'tx_lut',
                       th_lut: 'th_lut',
                       tw_lut: 'tw_lut'}

    for lut in lut_object_name.keys():
        name = lut_object_name[lut]
        self.constants[name] = PyTensor(self.name+name, lut.cpu().numpy().astype(dtype2nptype(lut_out_dtype)))

    score_q_min, score_q_max = bits2range(inp0.qbits, False)
    # self.params["box_shift_value"] = int(coord_shift)
    # self.params["box_shift_type"] = SHIFT_DTYPE
    self.params["score_threshold_value"] = linear_quantize_clip(score_thresh,
                                                                class_score_scale, class_score_zp, score_q_min, score_q_max).int().item()
    self.params['variance'] = [1, 1, 1, 1]
    out_type = [dtype2str(inp0.dtype), 'int16', 'uint16', 'uint16', 'uint16']
    out_scale = [class_score_scale, stat_coords_scale, 1, 1, 1]
    out_zerop = [class_score_zp, stat_coords_zp, 0, 0, 0]
    qinvariant_list = [False, False, True, True, True]
    for idx, out in enumerate(self.outputs):
        dtype = str2dtype(out_type[idx])
        qbits = dtype2bits(dtype)
        out.dtype = dtype
        out.scale = out_scale[idx]
        out.zerop = out_zerop[idx]
        out.qbits = qbits
        out.qinvariant = qinvariant_list[idx]
