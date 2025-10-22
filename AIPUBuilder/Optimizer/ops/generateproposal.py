# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch


def FAIL_LOG(result, log):
    if result:
        OPT_FATAL(log)


def check_bottom_nodes(node):
    if len(node.inputs) != 3:
        OPT_FATAL("input count must be 3")

    class_score = node.inputs[0].betensor
    box_encoding = node.inputs[1].betensor
    image_info = node.inputs[2].betensor
    FAIL_LOG(class_score.dim() != 4, 'the class confidence must have 4-dimention!')
    FAIL_LOG(box_encoding.dim() != 4, 'the box encodings must have 4-dimention!')
    FAIL_LOG(class_score.shape[1] != box_encoding.shape[1],
             'the class scores and box encoding must have the same total box num!')


def _get_box_score(class_score, box_encoding, score_thresh):
    max_detection_num = box_encoding.shape[0]
    max_class_num = class_score.shape[-1]
    crop_box_num = class_score.shape[0]
    box = torch.zeros((max_detection_num, 4), dtype=torch.float32, device=class_score.device)  # [300*80, 4]
    score = torch.zeros((max_detection_num), dtype=torch.float32, device=class_score.device)  # [300*80]
    box_num_perClass = torch.zeros([max_class_num], dtype=torch.float32, device=class_score.device)  # [80]

    outbox_idx = 0
    box_num_perClass_list = []
    class_label = []
    for class_idx in range(max_class_num):
        box_num_curClass = 0
        for box_idx in range(crop_box_num):
            if class_score[box_idx][class_idx] > score_thresh:
                if outbox_idx >= max_detection_num:
                    break
                if class_idx not in class_label:
                    class_label.append(class_idx)
                box[outbox_idx, :] = box_encoding[box_idx * max_class_num + class_idx, :]
                score[outbox_idx] = class_score[box_idx, class_idx]
                outbox_idx += 1
                box_num_curClass += 1
        if box_num_curClass != 0:
            box_num_perClass_list.append(box_num_curClass)
    box_num_perClass[:len(box_num_perClass_list)] = box_num_perClass_list
    total_class_num = len(class_label)
    class_label.extend((max_class_num - total_class_num) * [0])
    return box, score, box_num_perClass, class_label, total_class_num


def get_box_score(batch_class_score, batch_box_encoding, score_thresh):
    batch_size = batch_class_score.shape[0]
    max_detection_num = batch_box_encoding.shape[1]
    max_class_num = batch_class_score.shape[-1]
    box = torch.zeros((batch_size, max_detection_num, batch_box_encoding.shape[2]),
                      device=batch_class_score.device)
    score = torch.zeros((batch_size, max_detection_num), device=batch_class_score.device)
    box_num_perClass = torch.zeros((batch_size, max_class_num), device=batch_class_score.device)
    class_label = torch.zeros([batch_size, max_class_num], dtype=torch.int, device=batch_class_score.device)
    total_class_num = torch.ones([batch_size, 1], dtype=torch.int, device=batch_class_score.device)

    for i in range(batch_class_score.shape[0]):
        box[i], score[i], box_num_perClass[i], class_label[i], total_class_num[i] = _get_box_score(
            batch_class_score[i], batch_box_encoding[i], score_thresh)

    return box, score, box_num_perClass, class_label, total_class_num


def _filter_boxes(node, boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    if not node.quantized:
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
    else:
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
    # align with lib, only reserve the greater min_size box not greater_equal min_size
    keep = torch.where((ws > min_size) & (hs > min_size))[0]
    return keep


def NMS_F(boxes, scores, iou_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1+1) * (y2 - y1+1)
    #order = torch.argsort(torch.flatten(scores), descending=True)
    order = torch.linspace(0, len(torch.flatten(scores))-1,
                           steps=len(torch.flatten(scores)), device=scores.device).long()
    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        w = torch.maximum(torch.tensor([0], device=boxes.device), xx2 - xx1+1)
        h = torch.maximum(torch.tensor([0], device=boxes.device), yy2 - yy1+1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = torch.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def NMS_Q(box, score, iou_threshold=None):
    iou_thresh_shift = 11
    # currently area_shift is consistent with lib, and  will write to quantIR and will be modified
    area_shift = 15

    x0 = box[:, 0]
    y0 = box[:, 1]
    x1 = box[:, 2]
    y1 = box[:, 3]
    areas = (y1 - y0).type(torch.int32) * (x1 - x0).type(torch.int32)
    areas = areas >> area_shift
    # order = score.ravel().argsort()[::-1]

    # order = torch.argsort(torch.flatten(score), descending=True)
    order = torch.linspace(0, len(torch.flatten(score))-1, steps=len(torch.flatten(score)), device=score.device).long()
    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)

        xx0 = torch.maximum(x0[i], x0[order[1:]])
        yy0 = torch.maximum(y0[i], y0[order[1:]])
        xx1 = torch.minimum(x1[i], x1[order[1:]])
        yy1 = torch.minimum(y1[i], y1[order[1:]])
        w = torch.maximum(torch.tensor([0], device=box.device), xx1 - xx0)
        h = torch.maximum(torch.tensor([0], device=box.device), yy1 - yy0)

        inter = (w * h).type(torch.int32) >> area_shift
        union = areas[i] + areas[order[1:]] - inter
        inter_area_thresh = union * iou_threshold >> int(iou_thresh_shift)

        inds = torch.where(inter <= inter_area_thresh)[0]
        order = order[inds + 1]

    return keep


@op_register(OpType.GenerateProposals)
def generateproposals(self, *args):
    check_bottom_nodes(self)
    inp_tl = self.inputs
    class_score, box_encoding, image_info = inp_tl[0].betensor, inp_tl[1].betensor, inp_tl[2].betensor
    pre_nms_topn = self.get_param('pre_nms_topn')
    post_nms_topn = self.get_param('post_nms_topn')
    image_height = self.get_param('image_height')
    image_width = self.get_param('image_width')
    min_size = self.get_param('min_size')
    max_prop = post_nms_topn
    batch_num = class_score.shape[0]
    feat_stride = 16  # image_height / box_encoding.shape[1]
    num_anchors = 12  # class_score.shape[3]
    box_encoding = torch.reshape(box_encoding, (batch_num, -1, 4))  # [K*A, 4]
    total_box_num = class_score.shape[1]*class_score.shape[2]*class_score.shape[3]
    out_boxes = torch.zeros([batch_num, max_prop, 4], device=class_score.device)
    out_scores = torch.zeros([batch_num, max_prop], device=class_score.device)
    total_class_num = torch.ones([batch_num, 1], device=class_score.device)  # not used
    batch_index = torch.zeros([batch_num, max_prop, 1], device=class_score.device)
    ymin_batch, xmin_batch, ymax_batch, xmax_batch = [], [], [], []
    dev = class_score.device
    if not self.quantized:
        coords_stats = []
        box_stats = []
        _anchors = self.constants["weights"].betensor
        if _anchors.shape[0] != box_encoding.shape[1]:
            height, width = class_score.shape[1:3]
            shift_x = torch.arange(width, dtype=torch.float32, device=_anchors.device) * feat_stride
            shift_y = torch.arange(height, dtype=torch.float32, device=_anchors.device) * feat_stride

            shift_y, shift_x = torch.meshgrid(shift_x, shift_y)  # [width, height]
            shifts = torch.vstack((torch.flatten(shift_x), torch.flatten(shift_y),
                                   torch.flatten(shift_x), torch.flatten(shift_y))).permute(1, 0)
            K = shifts.shape[0]
            A = _anchors.shape[0]
            anchor_box = torch.reshape(_anchors, (1, A, 4)) + torch.reshape(shifts, (1, K, 4)).permute((1, 0, 2))
            anchor_box = torch.reshape(anchor_box, (K * A, 4))

            wa = (anchor_box[:, 2] - anchor_box[:, 0] + 1.0)
            ha = (anchor_box[:, 3] - anchor_box[:, 1] + 1.0)
            xcenter_a = (anchor_box[:, 0] + 0.5 * wa)
            ycenter_a = (anchor_box[:, 1] + 0.5 * ha)
            new_weights = torch.stack([wa, ha, xcenter_a, ycenter_a], axis=-1)
            self.constants["weights"].betensor = new_weights
        else:
            wa = _anchors[:, 0]
            ha = _anchors[:, 1]
            xcenter_a = _anchors[:, 2]
            ycenter_a = _anchors[:, 3]

        for batch in range(batch_num):
            tx, ty, tw, th = box_encoding[batch, :, 0].float(), box_encoding[batch,
                                                                             :, 1].float(), box_encoding[batch, :, 2].float(), box_encoding[batch, :, 3].float()

            w = torch.exp(tw) * wa[0:tw.shape[0]]
            h = torch.exp(th) * ha[0:th.shape[0]]
            ycenter = ty * ha[0:tw.shape[0]] + ycenter_a[0:ty.shape[0]]
            xcenter = tx * wa[0:tw.shape[0]] + xcenter_a[0:tx.shape[0]]

            # upper left:[ymin,xmin], lower right:[ymax,xmin]
            ymin = ycenter - h / 2.0
            xmin = xcenter - w / 2.0
            ymax = ycenter + h / 2.0
            xmax = xcenter + w / 2.0

            ymin = torch.clamp(ymin, 0, image_height - 1)
            xmin = torch.clamp(xmin, 0, image_width - 1)
            ymax = torch.clamp(ymax, 0, image_height - 1)
            xmax = torch.clamp(xmax, 0, image_width - 1)
            ymin_batch.append(ymin)
            xmin_batch.append(xmin)
            ymax_batch.append(ymax)
            xmax_batch.append(xmax)

            coords_stats.extend([ycenter_a, xcenter_a,
                                 ha, wa,
                                 ymin, ymax, xmin, xmax, ycenter, xcenter, h, w,
                                 ty * ha, tx * wa])
            box_stats.extend([ty, tx, th, tw, torch.exp(th), torch.exp(tw)])
        placeholders = [coords_stats, box_stats]
        placeholders_output = []

        for placeholder in placeholders:
            tensor_all = placeholder[0]
            for idx, tensor in enumerate(placeholder):
                tensor_all = tensor if idx == 0 else torch.cat((tensor_all, tensor), dim=0)
            placeholders_output.append(tensor_all)

        if len(self.placeholders) < 1:
            ph0 = PyTensor(self.name+"/coords", placeholders_output[0], dtype=Dtype.FP32)
            ph1 = PyTensor(self.name+"/box", placeholders_output[1], dtype=Dtype.FP32)
            self.placeholders.append(ph0)
            self.placeholders.append(ph1)
        self.placeholders[0].betensor = placeholders_output[0]
        self.placeholders[1].betensor = placeholders_output[1]

    else:
        if 'weights' in self.constants:
            self.constants.pop("weights")
        tw_lut = self.constants['wh_lut'].betensor
        th_lut = self.constants['wh_lut'].betensor
        ty_lut = self.constants['xy_lut'].betensor
        tx_lut = self.constants['xy_lut'].betensor

        anchor_scale = self.get_param("anchor_scale")
        shift = self.get_param("delta_shift")
        wa_q = self.constants['wa_q'].betensor
        ha_q = self.constants['ha_q'].betensor
        ycenter_a_q = self.constants['ycenter_a_q'].betensor
        xcenter_a_q = self.constants['xcenter_a_q'].betensor

        image_height = image_height * anchor_scale
        image_width = image_width * anchor_scale

        lut_in_bits = self.inputs[1].qbits
        in_is_signed = is_signed(self.inputs[1].dtype)
        hlut_out_bits = 16  # dtype2bits(self.get_constant('wh_lut').dtype)
        ylut_out_bits = 16  # dtype2bits(self.get_constant('xy_lut').dtype)
        hout_is_signed = is_signed(self.get_constant('wh_lut').dtype)
        yout_is_signed = is_signed(self.get_constant('xy_lut').dtype)

        for batch in range(batch_num):
            tx, ty, tw, th = box_encoding[batch, :, 0], box_encoding[batch,
                                                                     :, 1], box_encoding[batch, :, 2], box_encoding[batch, :, 3]

            lut_h = lookup_lut_powerof2(th, th_lut, lut_in_bits, in_is_signed, hlut_out_bits, hout_is_signed)
            lut_w = lookup_lut_powerof2(tw, tw_lut, lut_in_bits, in_is_signed, hlut_out_bits, hout_is_signed)
            lut_y = lookup_lut_powerof2(ty, ty_lut, lut_in_bits, in_is_signed, ylut_out_bits, yout_is_signed)
            lut_x = lookup_lut_powerof2(tx, tx_lut, lut_in_bits, in_is_signed, ylut_out_bits, yout_is_signed)

            h = (lut_h.to(dev).int() * (ha_q).to(dev).int()) >> shift
            w = (lut_w.to(dev).int() * (wa_q).to(dev).int()) >> shift
            cy = ((lut_y.to(dev).int() * (ha_q).to(dev).int()) >> shift).int() + (ycenter_a_q).to(dev).int()
            cx = ((lut_x.to(dev).int() * (wa_q).to(dev).int()) >> shift).int() + (xcenter_a_q).to(dev).int()

            ymin = cy - torch.div(h, 2, rounding_mode='trunc')
            xmin = cx - torch.div(w, 2, rounding_mode='trunc')
            ymax = cy + torch.div(h, 2, rounding_mode='trunc')
            xmax = cx + torch.div(w, 2, rounding_mode='trunc')

            ymin = torch.clamp(ymin, 0, image_height - anchor_scale)
            ymax = torch.clamp(ymax, 0, image_height - anchor_scale)
            xmin = torch.clamp(xmin, 0, image_width - anchor_scale)
            xmax = torch.clamp(xmax, 0, image_width - anchor_scale)
            ymin_batch.append(ymin)
            xmin_batch.append(xmin)
            ymax_batch.append(ymax)
            xmax_batch.append(xmax)

    for batch in range(batch_num):
        bboxes = torch.stack((xmin_batch[batch], ymin_batch[batch], xmax_batch[batch], ymax_batch[batch]), dim=-1)
    # remove predicted boxes with either height or width < min_size
        proposal_scores = torch.flatten(class_score[batch])
        order = torch.argsort(proposal_scores, descending=True)  # proposal_scores = [1*38*38*12, 1]
        order = order[:pre_nms_topn]
        bboxes = bboxes[order, :]
        proposal_scores = proposal_scores[order]
        keep = _filter_boxes(self, bboxes, min_size)

        bboxes = bboxes[keep, :]
        proposal_scores = proposal_scores[keep]

        iou_threshold = self.get_param('iou_threshold')
        if not self.quantized:
            keep = NMS_F(bboxes.float(), proposal_scores.float(), iou_threshold)
        else:
            keep = NMS_Q(bboxes, proposal_scores, int(iou_threshold))

        keep = keep[:post_nms_topn]

        out_boxes[batch, 0:len(keep), :] = bboxes[keep, :]
        out_scores[batch, :len(keep)] = proposal_scores[[keep]]

        total_class_num[batch] = len(keep)

    self.outputs[0].betensor = out_scores
    self.outputs[1].betensor = out_boxes
    self.outputs[2].betensor = batch_index
    self.outputs[3].betensor = total_class_num

    return [o.betensor for o in self.outputs]


@quant_register(OpType.GenerateProposals)
def generateproposals_quantize(self, *args):
    q_mode_bias = self.attrs["q_mode_bias"]
    q_mode_weight = self.attrs["q_mode_weight"]
    q_mode_activation = self.attrs["q_mode_activation"]
    if q_mode_weight != q_mode_bias:
        OPT_FATAL("Currently quantization mode of weight (q_mode_weight) and bias (q_mode_bias) must be the same!")
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_weight = self.attrs["q_bits_weight"]
    q_bits_bias = self.attrs["q_bits_bias"]
    q_bits_activation = self.attrs["q_bits_activation"]

    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    inp2 = self.inputs[2]

    class_scale, class_zp = inp0.scale, inp0.zerop
    box_encoding_scale, box_encoding_zp = inp1.scale, inp1.zerop

    coords = self.placeholders[0]
    box = self.placeholders[1]
    coords.qbits = max(16, q_bits_activation)
    coords.scale, coords.zerop, coords.qmin, coords.qmax, coords.dtype = get_linear_quant_params_from_tensor(
        coords, QuantMode.to_symmetric(q_mode_activation), coords.qbits, is_signed=True)
    coords.qinvariant = False
    stat_coords_scale, stat_coords_zp = coords.scale, coords.zerop
    stat_coords_scale = 2 ** torch.floor(torch.log2(torch.tensor(stat_coords_scale))).item()
    anchor_dtype = Dtype.INT16
    anchor_rmin, anchor_rmax = dtype2range(anchor_dtype)

    _anchors = self.constants["weights"].betensor
    wa = _anchors[:, 0]
    ha = _anchors[:, 1]
    xcenter_a = _anchors[:, 2]
    ycenter_a = _anchors[:, 3]
    ycenter_a_q = linear_quantize_clip(ycenter_a, stat_coords_scale, stat_coords_zp, anchor_rmin, anchor_rmax)
    xcenter_a_q = linear_quantize_clip(xcenter_a, stat_coords_scale, stat_coords_zp, anchor_rmin, anchor_rmax)
    ha_q = linear_quantize_clip(ha, stat_coords_scale, stat_coords_zp, anchor_rmin, anchor_rmax)
    wa_q = linear_quantize_clip(wa, stat_coords_scale, stat_coords_zp, anchor_rmin, anchor_rmax)

    constants_name = ['ycenter_a_q', 'xcenter_a_q', 'ha_q', 'wa_q']
    constants_list = [ycenter_a_q, xcenter_a_q, ha_q, wa_q]
    for name, a_q in zip(constants_name, constants_list):
        self.constants[name] = PyTensor(self.name+name, a_q, dtype=Dtype.INT16)
        self.constants[name].dtype = Dtype.INT16
        self.constants[name].ir_dtype = Dtype.INT16

    box.qbits = max(16, q_bits_activation)
    box.scale, box.zerop, box.qmin, box.qmax, box.dtype = get_linear_quant_params_from_tensor(
        box, QuantMode.to_symmetric(q_mode_activation), box.qbits, is_signed=True)
    box.qinvariant = False
    coord_scale = box.scale
    coord_zerop = box.zerop
    coord_shift = torch.floor(torch.log2(torch.tensor(coord_scale))).item()
    coord_scale = 2 ** coord_shift

    def get_lut(in_scale, in_zerop, var, box_scale, box_zerop, lut_in_dtype, lut_size_bits, lut_range_dtype, flag):
        lsteps = 2 ** lut_size_bits
        in_qmin, in_qmax = dtype2range(lut_in_dtype)
        if in_zerop == 0:
            in_qmax += 1
        lut_o_qmin, lut_o_qmax = dtype2range(lut_range_dtype)
        lut = linear_dequantize(torch.linspace(in_qmin, in_qmax, steps=lsteps), in_scale, in_zerop)
        lut = lut / var
        if flag:
            lut = torch.exp(lut)
        lut = linear_quantize_clip(lut, box_scale, box_zerop, lut_o_qmin, lut_o_qmax)
        return lut

    var_list = [1.0, 1.0, 1.0, 1.0]
    lut_in_dtype = inp1.dtype
    lut_size_bits = min(inp1.qbits, int(self.get_attrs('lut_items_in_bits')))
    lut_range_bits = max(self.attrs['q_bits_activation'], 16)
    lut_out_dtype = bits2dtype(lut_range_bits, True)
    ty_lut = get_lut(box_encoding_scale, box_encoding_zp,
                     var_list[1], coord_scale, coord_zerop, lut_in_dtype, lut_size_bits, lut_out_dtype, False)
    th_lut = get_lut(box_encoding_scale, box_encoding_zp,
                     var_list[3], coord_scale, coord_zerop, lut_in_dtype, lut_size_bits, lut_out_dtype, True)

    # lib needs 32bits ty_lut and th_lut, and ty_lut and th_lut actually save 16bits data.
    lut_object_name = {ty_lut: 'xy_lut', th_lut: 'wh_lut'}
    for lut in lut_object_name.keys():
        name = lut_object_name[lut]
        self.constants[name] = PyTensor(self.name+name, lut, dtype=Dtype.INT32)
        self.constants[name].dtype = Dtype.INT32
        self.constants[name].ir_dtype = Dtype.INT32

    feature_height = inp1.ir_shape[1]
    feature_width = inp1.ir_shape[2]
    image_height = self.get_param('image_height')
    image_width = self.get_param('image_width')
    self.params["anchor_scale"] = int(stat_coords_scale)
    self.params["min_size"] = int((self.get_param('min_size')-1) * stat_coords_scale)
    self.params['iou_scale'] = 2048
    self.params['iou_threshold'] = int(self.get_param('iou_threshold') * 2048)
    self.params["height_stride"] = (torch.ceil(torch.tensor(image_height/feature_height))
                                    * stat_coords_scale).int().item()
    self.params["width_stride"] = (torch.ceil(torch.tensor(image_width/feature_width)) * stat_coords_scale).int().item()
    self.params["delta_shift"] = int(coord_shift)
    self.params["roi_last_shift"] = int(coord_shift)

    ################################meet lib requirement and only generate to write to ir ###################################
    w = self.constants["weights"]
    fp_base_anchor = w.betensor
    fp_base_anchor[:, 2:] = fp_base_anchor[:, 2:] + 1
    quant_base_anchor = linear_quantize_clip(fp_base_anchor, stat_coords_scale,
                                             stat_coords_zp, anchor_rmin, anchor_rmax).type(torch.int16)
    quant_base_anchor = torch.flatten(quant_base_anchor).cpu()
    th = torch.flatten(th_lut).int()
    ty = torch.flatten(ty_lut).int()
    weights = torch.cat((quant_base_anchor, th, ty))
    w.scale, w.zerop, w.qmin, w.qmax = stat_coords_scale, stat_coords_zp, anchor_rmin, anchor_rmax
    w.betensor = weights
    w.dtype = anchor_dtype
    w.qbits = dtype2bits(anchor_dtype)
    w.qinvariant = False
    self.constants['anchors'] = PyTensor(
        self.name+'/anchors', quant_base_anchor.numpy().astype(dtype2nptype(Dtype.INT16)))
    self.constants['anchors'].dtype = Dtype.INT16
    self.constants['anchors'].ir_dtype = Dtype.INT16

    out_type = [dtype2str(inp0.dtype), 'uint16', 'uint16', 'uint16']
    out_scale = [class_scale, stat_coords_scale, stat_coords_scale, 1]
    out_zerop = [class_zp, stat_coords_zp, stat_coords_zp, 0]
    qinvariant_list = [False, False, False, True]
    for idx, out in enumerate(self.outputs):
        dtype = str2dtype(out_type[idx])
        qbits = dtype2bits(dtype)
        out.dtype = dtype
        out.scale = out_scale[idx]
        out.zerop = out_zerop[idx]
        out.qbits = qbits
        out.qinvariant = qinvariant_list[idx]
