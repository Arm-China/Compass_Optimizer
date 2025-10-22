# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
from AIPUBuilder.Optimizer.logger import *


def del_tensor_from_index(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)


def iou(x0, y0, x1, y1, box_id, box_index):
    #x0[box_id], x0[box_index]
    xmin_A = torch.min(x0[box_id], x1[box_id])
    xmax_A = torch.max(x0[box_id], x1[box_id])
    ymin_A = torch.min(y0[box_id], y1[box_id])
    ymax_A = torch.max(y0[box_id], y1[box_id])

    xmin_B = torch.min(x0[box_index], x1[box_index])
    xmax_B = torch.max(x0[box_index], x1[box_index])
    ymin_B = torch.min(y0[box_index], y1[box_index])
    ymax_B = torch.max(y0[box_index], y1[box_index])

    intersection_ymin = torch.max(ymin_A, ymin_B)
    intersection_xmin = torch.max(xmin_A, xmin_B)
    intersection_ymax = torch.min(ymax_A, ymax_B)
    intersection_xmax = torch.min(xmax_A, xmax_B)

    intersection_h = torch.max(intersection_ymax - intersection_ymin, torch.tensor(0).to(x0.device))
    intersection_w = torch.max(intersection_xmax - intersection_xmin, torch.tensor(0).to(x0.device))
    return intersection_w, intersection_h


def single_softnms(self, box, score, max_nms_box_num_per_class):
    nms_box = torch.zeros((max_nms_box_num_per_class, 4))
    nms_score = torch.zeros((max_nms_box_num_per_class))
    keep = torch.zeros((max_nms_box_num_per_class))
    device = box.device

    score_threshold = float(self.get_param('score_threshold'))
    iou_threshold = self.get_param('iou_threshold')
    areas_shift = self.params['areas_shift'] if self.quantized else 0

    y0 = box[:, 0]
    x0 = box[:, 1]
    y1 = box[:, 2]
    x1 = box[:, 3]

    if self.quantized:
        areas = torch.abs((y1 - y0) * (x1 - x0)).long()
        areas = areas >> areas_shift
        iou_thresh_shift = self.params['iou_thresh_shift']
        score = score.int()
    else:
        areas = torch.abs((y1 - y0) * (x1 - x0))
        iou_thresh_shift = 0
        score = score.float()

    current_box_num = box.shape[0]
    greater_score_thres_mask = score > score_threshold
    box_index = torch.arange(current_box_num, device=score.device)[greater_score_thres_mask]
    suppress_begin_index = torch.zeros_like(box_index, device=score.device)
    score_cand = score[greater_score_thres_mask].clone()
    selected_boxes_index = []

    keep_idx = 0
    while (keep_idx < max_nms_box_num_per_class and score_cand.numel() > 0):
        argmax_idx = torch.argmax(score_cand, dim=-1)
        box_id = box_index[argmax_idx]
        score_value = score_cand[argmax_idx]
        origin_score = score_value.clone()

        for j in range(suppress_begin_index[argmax_idx], len(selected_boxes_index)):
            bid = selected_boxes_index[j]
            w, h = iou(x0, y0, x1, y1, box_id, bid)
            inter = w * h
            if self.quantized:
                scale_lut = self.constants['gaussian_scale_lut'].betensor
                shift_lut = self.constants['gaussian_shift_lut'].betensor
                soft_nms_sigma_in_shift = self.params["soft_nms_sigma_in_shift"]
                inter = (inter).int()
                inter = inter >> areas_shift
                union = areas[box_id] + areas[bid] - inter
                offset = 2 ** soft_nms_sigma_in_shift - 1
                inter[union == 0] = 0
                union[union == 0] = 1
                ious = (inter * offset // union).long()
                score_value = (score_value * scale_lut[ious]).long() >> shift_lut[ious].int()
            else:
                union = areas[box_id] + areas[bid] - inter
                soft_nms_sigma = -0.5 / self.get_param('soft_nms_sigma')
                ious = inter / union
                if union == 0:
                    ious = torch.tensor(0, device=score.device)
                score_value = score_value * torch.exp(soft_nms_sigma * ious * ious)

            if score_value.float() <= score_threshold:
                break

        suppress_begin_index[argmax_idx] = len(selected_boxes_index)
        if origin_score == score_value:
            keep[keep_idx] = box_id
            nms_box[keep_idx, :] = box[box_id, :]
            nms_score[keep_idx] = score_value
            selected_boxes_index.append(box_id.item())
            keep_idx += 1
            score_cand = del_tensor_from_index(score_cand, argmax_idx)
            box_index = del_tensor_from_index(box_index, argmax_idx)
            suppress_begin_index = del_tensor_from_index(suppress_begin_index, argmax_idx)
            continue
        if score_value <= score_threshold:
            score_cand = del_tensor_from_index(score_cand, argmax_idx)
            box_index = del_tensor_from_index(box_index, argmax_idx)
            suppress_begin_index = del_tensor_from_index(suppress_begin_index, argmax_idx)
        else:
            score_cand[argmax_idx] = score_value

    boxNum_perclass = keep_idx
    if self.quantized:
        box_scale = self.params['scale_value']
        box_shift = self.params['shift_value']
        nms_box = nms_box.int() * box_scale >> box_shift
    nms_box = nms_box[0:boxNum_perclass, :]
    nms_score = nms_score[0:boxNum_perclass]
    keep = keep[0:boxNum_perclass]

    return nms_box, nms_score, boxNum_perclass, keep


def single_nms(self, box, score, max_nms_box_num):
    # keep = torchvision.ops.nms(box,score,iou_threshold)
    # outputs
    # iou_threshold =  self.params['iou_threshold']
    nms_box = torch.zeros((max_nms_box_num, 4))
    nms_score = torch.zeros((max_nms_box_num))
    keep = torch.zeros((max_nms_box_num))

    y0 = box[:, 0]
    x0 = box[:, 1]
    y1 = box[:, 2]
    x1 = box[:, 3]

    # it will set optional to False when parser add 'score_threshold' in future
    score_threshold = float(self.get_param('score_threshold', optional=True, default_value='-inf'))
    areas_shift = self.params['areas_shift'] if self.quantized else 0

    if self.quantized:
        iou_threshold = self.params['iou_threshold']
        iou_thresh_shift = self.params['iou_thresh_shift']
        box_scale = self.params['scale_value']
        box_shift = self.params['shift_value']
        areas = torch.abs((y1 - y0) * (x1 - x0)).int()
        areas = areas >> areas_shift
    else:
        iou_threshold = self.get_param('iou_threshold')
        areas = torch.abs((y1 - y0) * (x1 - x0))

    order = score[:].argsort(dim=-1, descending=True)  # descending order
    keep_idx = 0
    boxNum_perclass_single = 0
    device = x0[0].device
    while order.size()[0] > 0:
        i = order[0]
        if score[i].float() <= score_threshold:
            break
        boxNum_perclass_single += 1

        keep[keep_idx] = i
        nms_box[keep_idx, :] = box[i, :]
        nms_score[keep_idx] = score[i]

        keep_idx += 1
        if keep_idx >= max_nms_box_num:
            break

        # xx0 = torch.max(x0[i], x0[order[1:]])
        # yy0 = torch.max(y0[i], y0[order[1:]])
        # xx1 = torch.min(x1[i], x1[order[1:]])
        # yy1 = torch.min(y1[i], y1[order[1:]])
        # w = torch.max(torch.tensor(0.0).to(device), xx1 - xx0)
        # h = torch.max(torch.tensor(0.0).to(device), yy1 - yy0)
        w, h = iou(x0, y0, x1, y1, i, order[1:])
        inter = w * h
        if self.quantized:
            inter = (inter).int()
            inter = inter >> areas_shift

        union = areas[i] + areas[order[1:]] - inter

        inter_area_thresh = union * iou_threshold
        if self.quantized:
            inter_area_thresh = union * iou_threshold >> int(iou_thresh_shift)

        inds = torch.where(inter <= inter_area_thresh)[0]
        order = order[inds + 1]

    boxNum_perclass = boxNum_perclass_single
    if self.quantized:
        nms_box = nms_box.int() * box_scale >> box_shift
        pass
    nms_box = nms_box[0:boxNum_perclass, :]
    nms_score = nms_score[0:boxNum_perclass]
    keep = keep[0:boxNum_perclass]
    return nms_box, nms_score, boxNum_perclass, keep


@op_register(OpType.NMS)
def Nms(self, *args):
    out = self.outputs[:]
    # get bottom node
    if self.quantized and not is_float(self.inputs[0].dtype):
        batch_proposal_boxes = self.inputs[0].betensor.int() + self.inputs[0].zerop
    else:
        batch_proposal_boxes = self.inputs[0].betensor.float()
    # batch_boxNum_perClass and batch_total_class_num are not quantized, so their zerop is 0
    batch_boxNum_perClass = self.inputs[1].betensor
    batch_total_class_num = self.inputs[2].betensor
    # input batch_proposal_scores zerop pass to output score, it can omit zerop
    batch_proposal_scores = self.inputs[3].betensor

    batch_num = batch_proposal_boxes.shape[0]
    max_class_num = self.outputs[1].ir_shape[1]
    max_nms_box_num = self.outputs[0].ir_shape[1]

    # it will set optional to False when parser add 'method' and 'max_output_size' in future
    method = self.get_param('method', optional=True, default_value='HARD')
    max_output_size = self.get_param(
        'max_output_size', optional=True, default_value=max_nms_box_num)
    support_method_ = ['HARD', 'GAUSSIAN']
    if method not in support_method_:
        OPT_WARN(f"NMS op now only supports {str(support_method_)} method, but now method={method}, "
                 f"and Opt will use 'hard-nms' method to continue.")
        method = 'HARD'

    nms_func = {
        "HARD":  single_nms,
        "GAUSSIAN": single_softnms,
    }
    dev = self.inputs[0].betensor.device
    batch_nms_boxes = torch.zeros((batch_num, max_nms_box_num, 4), device=dev)
    batch_nms_scores = torch.zeros((batch_num, max_nms_box_num), device=dev)
    batch_nms_boxNum_perClass = torch.zeros((batch_num, max_class_num), device=dev)
    batch_keep = torch.zeros((batch_num, max_nms_box_num), device=dev)
    batch_proposal_scores = torch.reshape(
        batch_proposal_scores, [batch_num, -1])
    for idx_batch in range(batch_num):
        proposal_boxes = batch_proposal_boxes[idx_batch]
        boxNum_perclass = batch_boxNum_perClass[idx_batch]  # [5000]
        total_class_num = batch_total_class_num[idx_batch]  # [1]
        proposal_scores = batch_proposal_scores[idx_batch]

        idx_proposals = 0
        idx_keep = 0
        tot_cls = int(total_class_num[0]) if total_class_num.ndim > 0 else int(total_class_num)
        for idx_class in range(tot_cls):
            if idx_keep > max_nms_box_num:
                break
            box_num = int(boxNum_perclass[idx_class])
            if box_num == 0:
                continue
            boxes = proposal_boxes[idx_proposals: idx_proposals + box_num, :].reshape(box_num, 4)
            scores = proposal_scores[idx_proposals: idx_proposals + box_num]

            nms_box_, nms_score_, nms_boxNum_perClass_, keep_ = nms_func[method](self, boxes, scores, max_output_size)
            if idx_keep + nms_boxNum_perClass_ > max_nms_box_num:
                nms_boxNum_perClass_ = max_nms_box_num - idx_keep

            batch_nms_boxes[idx_batch][idx_keep: idx_keep + nms_boxNum_perClass_] = nms_box_[0:nms_boxNum_perClass_]
            batch_nms_scores[idx_batch][idx_keep: idx_keep +
                                        nms_boxNum_perClass_] = nms_score_[0:nms_boxNum_perClass_]
            batch_keep[idx_batch][idx_keep: idx_keep +
                                  nms_boxNum_perClass_] = keep_[0:nms_boxNum_perClass_]
            batch_nms_boxNum_perClass[idx_batch][idx_class] = nms_boxNum_perClass_

            idx_proposals = idx_proposals + box_num
            idx_keep = idx_keep + nms_boxNum_perClass_

    out[0].betensor = batch_nms_boxes
    out[1].betensor = batch_nms_boxNum_perClass
    out[2].betensor = batch_nms_scores
    out[3].betensor = batch_keep

    return [o.betensor for o in self.outputs]


def generate_gussi_lut(in_qmin, in_qmax, soft_nms_sigma):
    step = in_qmax - in_qmin + 1
    lut = torch.linspace(in_qmin, in_qmax, steps=step) / in_qmax
    lut = torch.exp(soft_nms_sigma * lut * lut)
    do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(lut,
                                                                                      mult_bits=16,
                                                                                      force_shift_positive=False)
    return do_scale, do_shift


@quant_register(OpType.NMS)
def Nms_quantize(self, *args):
    inp = self.inputs[:]
    out = self.outputs[:]
    iou_threshold = self.get_param('iou_threshold')
    self.params.pop('iou_threshold')
    center_point_box = self.get_param(
        'center_point_box', optional=True, default_value=0)
    # input scale
    box_encoding_scale, box_encoding_zp = inp[0].scale, inp[0].zerop
    box_per_class_scale, box_per_class_zp = inp[1].scale, inp[1].zerop
    total_class_scale, total_class_zp = inp[2].scale, inp[2].zerop
    class_scale, class_zp = inp[3].scale, inp[3].zerop

    # get the height and width of the input image.
    if 'image_height' in self.params and 'image_width' in self.params and center_point_box:
        height = self.get_param('image_height')
        width = self.get_param('image_width')
    else:
        height = box_encoding_scale
        width = box_encoding_scale
    box_shift = 15  # fixed value
    max_h_w = max((height, width))

    if max_h_w <= int(box_encoding_scale) or inp[0].extrema_max <= 1.0:
        box_scale, box_scale_type, box_shift, box_shift_type = \
            get_scale_approximation_params(max_h_w / box_encoding_scale,
                                           mult_bits=16,
                                           force_shift_positive=self.force_shift_positive)
        # box_scale = int(math.ceil(max_h_w / box_encoding_scale * 2**box_shift))
        box_encoding_scale = float(max_h_w)
    else:
        box_scale, box_scale_type, box_shift, box_shift_type = \
            get_scale_approximation_params(1 / box_encoding_scale,
                                           mult_bits=16,
                                           force_shift_positive=self.force_shift_positive)
        # box_scale = int(math.ceil((2 ** box_shift) / box_encoding_scale))
        box_encoding_scale = 1.0

    # it will set optional to False when parser add 'method' and 'score_threshold' in future
    method = self.get_param('method', optional=True, default_value='HARD')
    support_method_ = ['HARD', 'GAUSSIAN']
    if method not in support_method_:
        OPT_WARN(f"NMS op now only supports {str(support_method_)} method, but now method={method}, "
                 f"and Opt will use 'hard-nms' method to continue.")
        method = 'HARD'
    score_threshold = float(self.get_param('score_threshold', optional=True, default_value='-inf'))

    if method == 'GAUSSIAN':
        soft_nms_sigma = self.get_param('soft_nms_sigma')
        soft_nms_sigma = -0.5 / soft_nms_sigma
        soft_nms_sigma_in_shift = 8
        in_qmin, in_qmax = 0, 2**soft_nms_sigma_in_shift-1
        do_scale, do_shift = generate_gussi_lut(
            in_qmin, in_qmax, soft_nms_sigma)
        self.constants['gaussian_scale_lut'] = PyTensor(
            self.name + '/soft_nms_sigma_do_scale', do_scale, dtype=Dtype.UINT16)
        self.constants['gaussian_shift_lut'] = PyTensor(
            self.name + '/soft_nms_sigma_do_shift', do_shift, dtype=Dtype.UINT8)
        self.params["soft_nms_sigma_in_shift"] = soft_nms_sigma_in_shift

    iou_thresh_scale = 256
    iou_thresh_shift = 8
    self.params['iou_thresh_shift'] = int(iou_thresh_shift)
    # iout_thresh_scale: *2048>>11
    iou_threshold = int(iou_threshold * iou_thresh_scale)
    self.params["iou_threshold"] = int(iou_threshold)

    score_q_min, score_q_max = bits2range(inp[3].qbits, False)

    out[0].scale, out[0].zerop = box_encoding_scale, 0
    out[1].scale, out[1].zerop = 1., 0
    out[2].scale, out[2].zerop = class_scale, class_zp
    out[3].scale, out[3].zerop = 1., 0

    # set dtpye and qbits
    out[0].dtype = inp[0].dtype
    out[0].qbits = inp[0].qbits
    out[0].qinvariant = inp[0].qinvariant
    out[2].dtype = inp[3].dtype
    out[2].qbits = inp[3].qbits
    out[2].qinvariant = inp[3].qinvariant
    out[1].dtype = Dtype.UINT16
    out[1].qbits = 16
    out[1].qinvariant = True
    out[3].dtype = Dtype.UINT16
    out[3].qbits = 16
    out[3].qinvariant = True

    # int IR params
    self.params['areas_shift'] = 13
    if score_threshold == float('-inf'):
        score_min, score_max = dtype2range(out[2].dtype)
        self.params["score_threshold"] = score_min - 1
    else:
        self.params["score_threshold"] = linear_quantize_clip(score_threshold,
                                                              class_scale, class_zp, score_q_min, score_q_max).int().item()
    # self.params["iou_threshold_int16"] = int(iou_threshold)
    # self.params["box_scale_int16"] = int(box_scale)
    # self.params["box_shift_int16"] = int(box_shift)
    self.params["scale_value"] = int(box_scale)
    self.params["scale_type"] = box_scale_type
    self.params["shift_type"] = box_shift_type
    self.params["shift_value"] = int(box_shift)
