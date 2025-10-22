# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import *
import torch
import torch.nn.functional as F


'''
layer_id=32
layer_name=yolo_region
layer_type=Region
layer_bottom=[reshape/Reshape_0]
layer_bottom_shape=[[1,13,13,5,25]]
layer_bottom_type=[float32]
layer_top=[yolo_region_out_score,yolo_region_box,yolo_region_total_box_per_class,yolo_region_label_per_class,yolo_region_all_class]
layer_top_shape=[[1,5000],[1,5000,4],[1,20],[1,20],[1,1]]
layer_top_type=[float32,float32,float32,float32,float32]
anchors=[1.3221,1.73145,3.19275,4.00944,5.05587,8.09892,9.47112,4.84053,11.2364,10.0071]
box_per_grid=5
grid_height=13
grid_width=13
max_box_num=5000
obj_thresh=0.3
grid_compensate=true

'''


def check_and_pick_max_box_num(region, proposal, max_box_num=5000, coords_and_conf_num=5):

    if proposal[0].numel() > max_box_num:
        # (OPT_DEBUG('Region Op has %d proposal box(>%d), optimizer will pick the %d box num.'
        #            % (proposal[0].numel(), max_box_num, max_box_num)))
        scores = region[proposal[0], proposal[1], proposal[2], proposal[3] + coords_and_conf_num]
        descending_scores, indices = torch.sort(scores, descending=True)
        descending_scores = descending_scores[:5000]
        indices = indices[:5000]
        return (proposal[0][indices], proposal[1][indices], proposal[2][indices], proposal[3][indices])
    else:
        return proposal


def get_roi_one_batch(region,
                      threshold,
                      coords_and_conf_num,  # 5
                      anchors_t,
                      grid_height,
                      grid_width,
                      max_box_num,
                      roi_boxes,
                      roi_scores,
                      grid_compensate=False):

    proposal = torch.where(region[..., coords_and_conf_num:] > threshold)
    proposal = check_and_pick_max_box_num(region, proposal, max_box_num, coords_and_conf_num)
    gh_idxs, gw_idxs, box_idxs, cls_idxs = proposal[0],\
        proposal[1],\
        proposal[2],\
        proposal[3]
    proposal_class, num_proposal_class = torch.unique(cls_idxs, return_counts=True)
    total_class_num = proposal_class.numel()
    start_ps = torch.cumsum(num_proposal_class, dim=0) - num_proposal_class
    end_ps = torch.cumsum(num_proposal_class, dim=0)
    for t in range(total_class_num):
        idx = torch.where(cls_idxs == proposal_class[t])
        tx = region[gh_idxs[idx], gw_idxs[idx], box_idxs[idx], 0]
        ty = region[gh_idxs[idx], gw_idxs[idx], box_idxs[idx], 1]
        tw = region[gh_idxs[idx], gw_idxs[idx], box_idxs[idx], 2]
        th = region[gh_idxs[idx], gw_idxs[idx], box_idxs[idx], 3]

        if grid_compensate and grid_height % 2 == 0 and grid_width % 2 == 0:
            x = (gw_idxs[idx] - 0.5 + torch.sigmoid(tx)) / grid_width
            y = (gh_idxs[idx] - 0.5 + torch.sigmoid(ty)) / grid_height

        else:
            x = (gw_idxs[idx] + torch.sigmoid(tx)) / grid_width
            y = (gh_idxs[idx] + torch.sigmoid(ty)) / grid_height
        w = anchors_t[2*box_idxs[idx] + 0] * torch.exp(tw) / grid_width
        h = anchors_t[2*box_idxs[idx] + 1] * torch.exp(th) / grid_height

        ymin = torch.clamp(y - h / 2., 0., 1.)
        xmin = torch.clamp(x - w / 2., 0., 1.)
        ymax = torch.clamp(y + h / 2., 0., 1.)
        xmax = torch.clamp(x + w / 2., 0., 1.)

        roi_box = torch.stack([ymin, xmin, ymax, xmax], axis=-1)
        roi_boxes[start_ps[t]:end_ps[t], :] = roi_box
        roi_scores[start_ps[t]:end_ps[t]] = region[gh_idxs[idx],
                                                   gw_idxs[idx], box_idxs[idx], coords_and_conf_num+cls_idxs[idx]]

    return roi_boxes, roi_scores, num_proposal_class, proposal_class, total_class_num


def get_roi_one_batch_quant(self, region,
                            threshold,
                            coords_and_conf_num,  # 5
                            anchors_t,
                            xy_sigmoid_t,
                            wh_exp_t,
                            grid_col_shift,
                            grid_row_shift,
                            grid_h_scale,
                            grid_h_shift,
                            grid_w_scale,
                            grid_w_shift,
                            anchor_exp_h_shift,
                            anchor_exp_w_shift,
                            max_box_num,
                            roi_boxes,
                            roi_scores,
                            xy_in_bits=8,
                            wh_in_bits=8):
    xy_sigmoid_lut = xy_sigmoid_t.betensor
    xy_in_is_signed = True
    xy_out_is_signed = is_signed(xy_sigmoid_t.dtype)
    xy_out_bits = dtype2bits(xy_sigmoid_t.dtype)

    wh_exp_lut = wh_exp_t.betensor
    wh_in_is_signed = True
    wh_out_is_signed = is_signed(wh_exp_t.dtype)
    wh_out_bits = dtype2bits(wh_exp_t.dtype)

    proposal = torch.where(region[..., coords_and_conf_num:] > threshold)
    proposal = check_and_pick_max_box_num(region, proposal, max_box_num, coords_and_conf_num)
    gh_idxs, gw_idxs, box_idxs, cls_idxs = proposal[0], \
        proposal[1], \
        proposal[2], \
        proposal[3]

    proposal_class, num_proposal_class = torch.unique(cls_idxs, return_counts=True)
    total_class_num = proposal_class.numel()
    start_ps = torch.cumsum(num_proposal_class, dim=0) - num_proposal_class
    end_ps = torch.cumsum(num_proposal_class, dim=0)
    for t in range(total_class_num):
        idx = torch.where(cls_idxs == proposal_class[t])
        tx = region[gh_idxs[idx], gw_idxs[idx], box_idxs[idx], 0]
        ty = region[gh_idxs[idx], gw_idxs[idx], box_idxs[idx], 1]
        tw = region[gh_idxs[idx], gw_idxs[idx], box_idxs[idx], 2]
        th = region[gh_idxs[idx], gw_idxs[idx], box_idxs[idx], 3]

        x_lut = lookup_lut_powerof2(tx, xy_sigmoid_lut, xy_in_bits, xy_in_is_signed,
                                    xy_out_bits, xy_out_is_signed).int()
        y_lut = lookup_lut_powerof2(ty, xy_sigmoid_lut, xy_in_bits, xy_in_is_signed,
                                    xy_out_bits, xy_out_is_signed).int()
        w_lut = lookup_lut_powerof2(tw, wh_exp_lut, wh_in_bits, wh_in_is_signed, wh_out_bits, wh_out_is_signed).int()
        h_lut = lookup_lut_powerof2(th, wh_exp_lut, wh_in_bits, wh_in_is_signed, wh_out_bits, wh_out_is_signed).int()

        x = (gw_idxs[idx] << grid_col_shift).int() + x_lut.int()
        x = (x * grid_w_scale).int() >> grid_w_shift
        y = (gh_idxs[idx] << grid_row_shift).int() + y_lut.int()
        y = (y * grid_h_scale).int() >> grid_h_shift

        wh_exp_h_shift = self.get_param('wh_exp_shift')
        wh_exp_h_scale = self.get_param('wh_exp_scale')
        w1 = (anchors_t[(2*box_idxs[idx].long() + 0)] * grid_w_scale).int() >> grid_w_shift
        # w2 = wh_exp_lut[tw.long() + 128].int()*wh_exp_h_scale>>wh_exp_h_shift
        w2 = w_lut*wh_exp_h_scale >> wh_exp_h_shift
        w = (w1 * w2).int() >> anchor_exp_w_shift

        h1 = (anchors_t[(2*box_idxs[idx].long() + 1)] * grid_h_scale).int() >> grid_h_shift
        # h2 = wh_exp_lut[th.long() + 128].int()*wh_exp_h_scale>>wh_exp_h_shift
        h2 = h_lut * wh_exp_h_scale >> wh_exp_h_shift
        h = (h1 * h2).int() >> anchor_exp_h_shift

        ymin = torch.clamp(y - (h >> 1), 0, 2**grid_row_shift)
        xmin = torch.clamp(x - (w >> 1), 0, 2**grid_col_shift)
        ymax = torch.clamp(y + (h >> 1), 0, 2**grid_row_shift)
        xmax = torch.clamp(x + (w >> 1), 0, 2**grid_col_shift)
        roi_box = torch.stack([ymin, xmin, ymax, xmax], axis=-1)
        roi_boxes[start_ps[t]:end_ps[t], :] = roi_box
        roi_scores[start_ps[t]:end_ps[t]] = region[gh_idxs[idx], gw_idxs[idx],
                                                   box_idxs[idx], coords_and_conf_num+cls_idxs[idx]].int()
    return roi_boxes, roi_scores, num_proposal_class, proposal_class, total_class_num


@op_register(OpType.Region)
def region(self, *args):
    '''
    input shape = [batch, grid_h, grid_w, box_per_grid, (coord(4) + confidence(1) + class_num(20/80))]
    like in yolov2_416, the region op input shape is [batch, 13, 13, 5, 25]
    '''
    inp = torch.clone(self.inputs[0].betensor)

    dev = inp.device
    # OPT_INFO('shape:%s, tensor:%s'%(self.outputs[0].betensor.shape, self.outputs[0].betensor))

    obj_thresh = self.get_param('obj_thresh')
    # there are not max_box_num params in some IR, so we can read it from output or input shape
    default_max_box_num = self.outputs[1].ir_shape[1]
    max_box_num = self.get_param('max_box_num', optional=True, default_value=default_max_box_num)
    default_grid_height = self.inputs[0].ir_shape[-4]
    default_grid_width = self.inputs[0].ir_shape[-3]
    defalut_box_per_grid = self.inputs[0].ir_shape[-2]
    grid_height = self.get_param('grid_height', optional=True, default_value=default_grid_height)
    grid_width = self.get_param('grid_width', optional=True, default_value=default_grid_width)
    box_per_grid = self.get_param('box_per_grid', optional=True, default_value=defalut_box_per_grid)
    grid_compensate = self.get_param('grid_compensate', optional=True, default_value=True)
    coords_and_conf_num = 5  # 4 + 1
    class_num = inp.shape[4] - coords_and_conf_num
    batch_size = inp.shape[0]
    roi_score = torch.zeros([batch_size, max_box_num])
    roi_boxes = torch.zeros([batch_size, max_box_num, 4])
    roi_box_num_perclass = torch.zeros([batch_size, class_num])
    roi_label_perclass = torch.zeros([batch_size, class_num])
    roi_total_class_num = torch.zeros([batch_size, 1])

    if not self.quantized:
        # prepare params
        anchors = self.get_param('anchors')
        anchors_t = torch.tensor(anchors).to(dev)

        # step1: sigmoid(confidence)
        inp[..., 4:5] = torch.sigmoid(inp[..., 4:5])

        # step2: softmax(class_score) * confidence
        inp[..., 5:] = inp[..., 4:5] * F.softmax(inp[..., 5:], dim=-1)

        # step3: get class_score > obj_thresh and get bbox coordinate
        for bs in range(batch_size):
            ret = get_roi_one_batch(inp[bs, ...],
                                    obj_thresh,
                                    coords_and_conf_num,
                                    anchors_t,
                                    grid_height,
                                    grid_width,
                                    max_box_num,
                                    roi_boxes[bs],
                                    roi_score[bs],
                                    grid_compensate)
            roi_boxes[bs], roi_score[bs] = ret[0], ret[1]
            roi_box_num_perclass[bs, :ret[2].shape[0]] = ret[2]
            roi_label_perclass[bs, :ret[3].shape[0]] = ret[3]
            roi_total_class_num[bs] = torch.tensor(ret[4])
    else:
        qanchors = self.get_constant('qanchors_lut').betensor
        score_softmax_lut = self.get_constant('score_softmax_lut').betensor
        conf_sigmoid_lut = self.get_constant('conf_sigmoid_lut').betensor
        xy_sigmoid_t = self.get_constant('xy_sigmoid_lut')
        wh_exp_t = self.get_constant('wh_exp_lut')

        inp_t = self.inputs[0]
        raw_score = inp[..., 5:]
        score_max = torch.max(raw_score, dim=-1, keepdim=True)
        score = raw_score - score_max[0] + 2**inp_t.qbits - 1  # [-128, 127]
        score = torch.reshape(score, [-1])
        exp_score = lookup_lut_powerof2(score, score_softmax_lut, inp_t.qbits, False, dtype2bits(self.constants["score_softmax_lut"].dtype),
                                        is_signed(self.constants["score_softmax_lut"].dtype)).int()
        exp_score = torch.reshape(exp_score, raw_score.shape)
        # exp_score = score_softmax_lut[score.long() + 128 + 127] #[0, 255] => 2^20
        confidence = inp[..., 4:5]
        in_is_signed = is_signed(inp_t.dtype)
        lut_in_bits = inp_t.qbits
        lut_out_bits = dtype2bits(self.constants['conf_sigmoid_lut'].dtype)
        out_is_signed = is_signed(self.constants['conf_sigmoid_lut'].dtype)
        sigmoid_confidence = lookup_lut_powerof2(confidence, conf_sigmoid_lut,
                                                 lut_in_bits, in_is_signed, lut_out_bits, out_is_signed).int()

        conf_sigmoid_shift = self.get_param('conf_sigmoid_shift')
        sum_score = torch.sum(exp_score, dim=-1, keepdim=True)
        conf_softmax_score = (exp_score.long() * sigmoid_confidence.long()) >> conf_sigmoid_shift
        new_score_threshold = (sum_score * obj_thresh).long() >> 8
        invalid_score_mask = conf_softmax_score <= new_score_threshold
        conf_softmax_score[invalid_score_mask] = 0
        factor = 2 ** inp_t.qbits - 1
        conf_softmax_score = torch.div(conf_softmax_score * factor, sum_score, rounding_mode='trunc')
        inp[..., 5:] = conf_softmax_score

        anchor_exp_h_shift = self.get_param('anchors_exp_h_shift')
        anchor_exp_w_shift = self.get_param('anchors_exp_w_shift')
        grid_col_shift = self.get_param('col_shift')
        grid_row_shift = self.get_param('row_shift')
        grid_h_scale = self.get_param('grid_h_scale')
        grid_h_shift = self.get_param('grid_h_shift')
        grid_w_scale = self.get_param('grid_w_scale')
        grid_w_shift = self.get_param('grid_w_shift')
        for bs in range(batch_size):
            ret = get_roi_one_batch_quant(self, inp[bs, ...],
                                          obj_thresh,
                                          coords_and_conf_num,  # 5
                                          qanchors,
                                          xy_sigmoid_t,
                                          wh_exp_t,
                                          grid_col_shift,
                                          grid_row_shift,
                                          grid_h_scale,
                                          grid_h_shift,
                                          grid_w_scale,
                                          grid_w_shift,
                                          anchor_exp_h_shift,
                                          anchor_exp_w_shift,
                                          # coord_scale,
                                          max_box_num,
                                          roi_boxes[bs],
                                          roi_score[bs],
                                          xy_in_bits=inp_t.qbits,
                                          wh_in_bits=inp_t.qbits)
            roi_boxes[bs], roi_score[bs] = ret[0], ret[1]
            roi_box_num_perclass[bs, :ret[2].shape[0]] = ret[2]
            roi_label_perclass[bs, :ret[3].shape[0]] = ret[3]
            roi_total_class_num[bs] = torch.tensor(ret[4])

    self.outputs[0].betensor = roi_score
    self.outputs[1].betensor = roi_boxes
    self.outputs[2].betensor = roi_box_num_perclass
    self.outputs[3].betensor = roi_label_perclass
    self.outputs[4].betensor = roi_total_class_num

    return [o.betensor for o in self.outputs]


@quant_register(OpType.Region)
def quantize_region(self, *args):
    """
    :param self:
    :param args:
    :return:
    """

    # prepare parameters for quantize
    inp = self.inputs[0]
    dev = inp.betensor.device
    q_mode_out = self.attrs['q_mode_activation']
    q_bits_out = self.attrs['q_bits_activation']
    qout_min, qout_max = bits2range(q_bits_out, True)
    # update quantize precision for lut
    # [16, 32, 16, 16, 16, 16, 15] = [conf_sigmoid_shift, score_softmax_dtype_bits, anchor_dtype_bits,
    # conf_sigmoid_dtype_bits, bbox_xy_sigmoid_dtype_bits, bbox_wh_exp_dtype_bits, grid_shift]
    extra_params = self.get_attrs('extra_params', optional=True, default_value=[0, 15, 32, 16, 16, 16, 16, 15])
    _quantize_params = {
        'conf_sigmoid_shift': extra_params[1],
        'score_softmax_dtype': [extra_params[2], False],  # uint32
        'anchor_dtype': [extra_params[3], False],  # 'uint16',
        'conf_sigmoid_dtype': [extra_params[4], False],  # 'uint16',
        'bbox_xy_sigmoid_dtype': [extra_params[5], True],  # 'int16',
        'bbox_wh_exp_dtype': [extra_params[6], False],  # 'uint16',
    }
    for i, kv in enumerate(_quantize_params.items()):
        if i == 0:
            continue
        obits = max(kv[1][0], q_bits_out)
        dtype = bits2dtype(obits, kv[1][1])
        _quantize_params.update({kv[0]: [obits, kv[1][1], dtype]})

    # self.params['obj_thresh'] = int(self.params['obj_thresh'] * 255) #(2 ** inp.qbits - 1)
    self.params['obj_thresh'] = int(self.params['obj_thresh'] * (2 ** inp.qbits - 1))

    # step 1: get coord scale/shift for boundingbox regression
    # little trick
    box_signed = True
    box_qbits = 16
    _, box_qmax = bits2range(box_qbits, box_signed)
    coord_max = self.params['grid_width']
    coord_qmin, coord_qmax = bits2range(*(_quantize_params['anchor_dtype'][:2]))
    coord_shift = torch.floor(torch.log2(torch.tensor(coord_qmax / coord_max)))
    while 2 ** coord_shift > box_qmax:
        coord_shift -= 1
    coord_scale = 2 ** coord_shift

    self.params['col_shift'] = coord_shift.int().item()
    self.params['row_shift'] = coord_shift.int().item()
    self.attrs['coord_scale'] = coord_scale.item()

    # step 2: get quantized_grid_width and quantized_grid_height
    # grid_shift = 15
    grid_shift = extra_params[7]
    grid_scale = 2 ** grid_shift

    self.params['grid_w_shift'] = grid_shift
    self.params['grid_h_shift'] = grid_shift
    self.params['grid_w_scale'] = int(grid_scale / self.params['grid_width'])
    self.params['grid_h_scale'] = int(grid_scale / self.params['grid_height'])

    # step 3: quantize anchor to get quantized_anchor
    f_anchors = torch.tensor(self.get_param('anchors'), device=dev)
    anchors_tensor = PyTensor(self.name+'_anchors', f_anchors, dtype=Dtype.FP32)
    anchors_tensor.__setattr__('max', f_anchors.max().item())
    anchors_tensor.__setattr__('min', f_anchors.min().item())
    # anchor_quant_qbits = _quantize_params['default_coord_qbits']
    anchor_bits, out_signed = _quantize_params['anchor_dtype'][0], _quantize_params['anchor_dtype'][1]
    a_rmin, a_rmax = dtype2range(_quantize_params['anchor_dtype'][2])
    anchor_param = get_linear_quant_params_from_tensor(anchors_tensor, q_mode_out, anchor_bits, out_signed)
    anchor_exp_shift = torch.floor(torch.log2(anchor_param[0]))
    q_anchors = linear_quantize_clip(f_anchors, 2 ** anchor_exp_shift, 0, a_rmin, a_rmax)

    self.params['anchors_exp_h_shift'] = anchor_exp_shift.int().item()
    self.params['anchors_exp_w_shift'] = anchor_exp_shift.int().item()

    # step 4: get lut for score/confidence
    iqmin, iqmax = dtype2range(inp.dtype)
    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    f_range = linear_dequantize(torch.linspace(iqmin, iqmax, steps=lsteps), inp.scale, inp.zerop)
    # f_range = linear_dequantize(torch.arange(inp.qmin, inp.qmax + 1, device=dev), inp.scale, inp.zerop)
    f_sigmoid_lut = torch.sigmoid(f_range)

    scale_thtw_out = (65535/torch.tensor(6.39).exp())
    f_exp_range = linear_dequantize(torch.linspace(iqmin, iqmax, steps=lsteps), inp.scale, inp.zerop)
    # f_exp_range = linear_dequantize(torch.arange(inp.qmin, inp.qmax + 1, device=dev), inp.scale, inp.zerop)
    f_exp_lut = torch.exp(f_exp_range)
    # 4.1 softmax exp 32bit lut
    # score_softmax_quant_dtype = str2dtype(_quantize_params['score_softmax_dtype'])
    # so we allow to accumulate less than 2**12 items with a 32bit accumulator
    max_val = torch.tensor(1 << (32 - 12), dtype=torch.float, device=dev)
    max_inp = linear_quantize_clip(torch.log(max_val), inp.scale, inp.zerop,
                                   torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max)
    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    softmax_exp_lut = linear_dequantize(torch.linspace(0, 2**inp.qbits-1, steps=lsteps,
                                                       device=dev) + max_inp - 2**inp.qbits, inp.scale, inp.zerop)
    score_softmax_lut = torch.exp(softmax_exp_lut).round().clamp(0.0, max_val.item())

    # 4.2 sigmoid 16bit lut
    conf_sigmoid_qmin, conf_sigmoid_qmax = bits2range(*(_quantize_params['conf_sigmoid_dtype'][:2]))
    conf_sigmoid_shift = _quantize_params['conf_sigmoid_shift']
    self.params['conf_sigmoid_shift'] = conf_sigmoid_shift
    conf_sigmoid_lut = linear_quantize_clip(f_sigmoid_lut, 2 ** conf_sigmoid_shift,
                                            0, conf_sigmoid_qmin, conf_sigmoid_qmax)

    # 4.3 bboundingbox regression sigmoid/exp 16bit lut
    xy_qmin, xy_qmax = bits2range(*(_quantize_params['bbox_xy_sigmoid_dtype'][:2]))
    grid_compensate = self.get_param('grid_compensate', optional=True, default_value=False)

    # for caffe yolo416 which need compensate
    if grid_compensate and self.params['grid_height'] % 2 == 0 and self.params['grid_width'] % 2 == 0:
        xy_sigmoid_lut = f_sigmoid_lut - 0.5
    else:
        xy_sigmoid_lut = f_sigmoid_lut
    xy_sigmoid_lut = linear_quantize_clip(xy_sigmoid_lut, coord_scale, 0, xy_qmin, xy_qmax)

    wh_qmin, wh_qmax = bits2range(*(_quantize_params['bbox_wh_exp_dtype'][:2]))
    wh_exp_lut = linear_quantize_clip(f_exp_lut, scale_thtw_out, 0, wh_qmin, wh_qmax)

    xy_scale, xy_scale_type, xy_shift, xy_shift_type = \
        get_scale_approximation_params(coord_scale/scale_thtw_out, mult_bits=16,
                                       force_shift_positive=self.force_shift_positive)
    self.params['wh_exp_shift'] = int(xy_shift)
    self.params['wh_exp_scale'] = int(xy_scale)

    # 4.4 update weight constant data
    _lut_nptype_dict = {
        'anchor': _quantize_params['anchor_dtype'][2],
        'score': _quantize_params['score_softmax_dtype'][2],
        'conf': _quantize_params['conf_sigmoid_dtype'][2],
        'xy': _quantize_params['bbox_xy_sigmoid_dtype'][2],
        'wh': _quantize_params['bbox_wh_exp_dtype'][2],
    }

    self.constants['qanchors_lut'] = PyTensor(
        self.name+'/anchor_lut', q_anchors, dtype=_lut_nptype_dict['anchor'])
    self.constants['score_softmax_lut'] = PyTensor(
        self.name+'/score_softmax_lut', score_softmax_lut, dtype=_lut_nptype_dict['score'])
    self.constants['conf_sigmoid_lut'] = PyTensor(
        self.name+'/conf_sigmoid_lut', conf_sigmoid_lut, dtype=_lut_nptype_dict['conf'])
    self.constants['xy_sigmoid_lut'] = PyTensor(
        self.name+'/xy_sigmoid_lut', xy_sigmoid_lut, dtype=_lut_nptype_dict['xy'])
    self.constants['wh_exp_lut'] = PyTensor(
        self.name+'/wh_exp_lut', wh_exp_lut, dtype=_lut_nptype_dict['wh'])

    # step5. update the outputs tensor quantize parameters

    # _oscale_list = [inp.scale, coord_scale.item(), inp.scale, 1.0, 1.0]
    _oscale_list = [2**inp.qbits-1, coord_scale.item(), 1.0, 1.0, 1.0]
    _ozerop_list = [inp.zerop, 0, 0, 0, 0]
    _osigned_list = [False, box_signed, False, True, False]
    _act_qbits_list = [inp.qbits, box_qbits, 16, 16, 16]
    # _act_qbits_list = [max(q_bits_out, b) for b in _qbits_list]
    _dtype_list = [bits2dtype(b, s) for b, s in zip(_act_qbits_list, _osigned_list)]

    for i, o in enumerate(self.outputs):
        o.scale = _oscale_list[i]
        o.zerop = _ozerop_list[i]
        o.dtype = _dtype_list[i]
        o.qbits = _act_qbits_list[i]
    self.outputs[0].qinvariant = inp.qinvariant
    self.outputs[1].qinvariant = inp.qinvariant
    self.outputs[2].qinvariant = True
    self.outputs[3].qinvariant = True
    self.outputs[4].qinvariant = True

    # step6 pop usless param in quant IR
    self.params.pop('anchors')
