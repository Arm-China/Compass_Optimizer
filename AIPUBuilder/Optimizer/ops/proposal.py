# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *


def apply_box_deltas(self, config, score, deltas):
    dev = score.device
    wa = self.constants['wa'].betensor
    ha = self.constants['ha'].betensor
    ya = self.constants['ycenter'].betensor
    xa = self.constants['xcenter'].betensor

    if self.quantized:
        tw_lut = self.constants['tw_lut'].betensor
        th_lut = self.constants['th_lut'].betensor
        ty_lut = self.constants['ty_lut'].betensor
        tx_lut = self.constants['tx_lut'].betensor

        shift = self.params["box_shift_value"]
        image_height = self.get_param('height')
        image_width = self.get_param('width')

        ty = deltas[:, :, 0]
        tx = deltas[:, :, 1]
        th = deltas[:, :, 2]
        tw = deltas[:, :, 3]

        lut_in_bits = self.inputs[1].qbits
        lut_out_bits = dtype2bits(self.get_constant('ty_lut').dtype)
        in_is_signed = is_signed(self.inputs[1].dtype)
        out_is_signed = is_signed(self.get_constant('ty_lut').dtype)

        lut_h = lookup_lut_powerof2(th, th_lut, lut_in_bits, in_is_signed, lut_out_bits, out_is_signed)
        lut_w = lookup_lut_powerof2(tw, tw_lut, lut_in_bits, in_is_signed, lut_out_bits, out_is_signed)
        lut_y = lookup_lut_powerof2(ty, ty_lut, lut_in_bits, in_is_signed, lut_out_bits, out_is_signed)
        lut_x = lookup_lut_powerof2(tx, tx_lut, lut_in_bits, in_is_signed, lut_out_bits, out_is_signed)

        h = (lut_h.to(dev).int() * (ha).to(dev).int()) >> shift
        w = (lut_w.to(dev).int() * (wa).to(dev).int()) >> shift
        cy = ((lut_y.to(dev).int() * (ha).to(dev).int()) >> shift).int() + (ya).to(dev).int()
        cx = ((lut_x.to(dev).int() * (wa).to(dev).int()) >> shift).int() + (xa).to(dev).int()

        ymin = cy - h / 2.
        xmin = cx - w / 2.
        ymax = cy + h / 2.
        xmax = cx + w / 2.

        ymin = torch.clamp(ymin, 0, image_height)
        ymax = torch.clamp(ymax, 0, image_height)
        xmin = torch.clamp(xmin, 0, image_width)
        xmax = torch.clamp(xmax, 0, image_width)

        box = torch.stack([ymin, xmin, ymax, xmax], dim=2)
        # box_q32  = torch.stack([ymin_q32, xmin_q32, ymax_q32, xmax_q32], dim=2)
        # box_q16  = torch.clip(box_q32, 0, 32767) # clip to 0~1.0

        return box, score[:, :, 1]
    else:
        height = self.get_param('height')
        width = self.get_param('width')

        dy = deltas[:, :, 0].float().clone()
        dx = deltas[:, :, 1].float().clone()
        dh = deltas[:, :, 2].float().clone()
        dw = deltas[:, :, 3].float().clone()

        dy /= config.STD_DIV[0]
        dx /= config.STD_DIV[1]
        dh /= config.STD_DIV[2]
        dw /= config.STD_DIV[3]

        # adjust achors size and position
        ycenter = dy * ha + ya
        xcenter = dx * wa + xa
        h = torch.exp(dh) * ha
        w = torch.exp(dw) * wa

        ymin = ycenter - h / 2.
        xmin = xcenter - w / 2.
        ymax = ycenter + h / 2.
        xmax = xcenter + w / 2.

        ymin = torch.clamp(ymin, 0, height)
        xmin = torch.clamp(xmin, 0, width)
        ymax = torch.clamp(ymax, 0, height)
        xmax = torch.clamp(xmax, 0, width)

        box = torch.stack([ymin, xmin, ymax, xmax], dim=2)

        # print("box(after apply delta) before clip:",torch.min(box), torch.max(box))
        # clip in normalized size[0~1.0]
        # box[:,:,0][box[:,:,0] < 0]      = 0
        # box[:,:,1][box[:,:,1] < 0]      = 0
        # box[:,:,2][box[:,:,2] < 0]      = 0
        # box[:,:,3][box[:,:,3] < 0]      = 0
        # box[:,:,0][box[:,:,0] > 1]      = 1
        # box[:,:,1][box[:,:,1] > 1]      = 1
        # box[:,:,2][box[:,:,2] > 1]      = 1
        # box[:,:,3][box[:,:,3] > 1]      = 1

        coords_stats = [ycenter, xcenter, h, w,
                        ymin, ymax, xmin, xmax, ycenter, xcenter, h, w,
                        dy * h, dx * w]
        box_stats = [dy, dx, dh, dw, torch.exp(dh), torch.exp(dw)]
        placeholders = [coords_stats, box_stats]
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
            self.placeholders.append(ph0)
            self.placeholders.append(ph1)
        self.placeholders[0].betensor = placeholders_output[0]
        self.placeholders[1].betensor = placeholders_output[1]

    return box, score[:, :, 1]


def _get_box_score(node, class_score, box_encoding, score_thresh):
    max_prop_num = node.outputs[0].ir_shape[1]
    arg_score = torch.where(class_score > score_thresh)[0].long()
    box = box_encoding[arg_score, :][:max_prop_num, :]
    if node.quantized:
        box = torch.round(box).int()
    score = class_score[arg_score][:max_prop_num]
    outbox_idx = box.shape[0]
    return box, score, outbox_idx


def get_box_score(node, batch_class_score, batch_box_encoding):
    if not node.quantized:
        score_thresh = node.get_param('score_threshold')
        batch_class_score = batch_class_score.float()
    else:
        score_thresh = node.get_param('score_threshold_value')
    max_prop_num = node.outputs[0].ir_shape[1]
    dev = node.outputs[0].betensor.device
    box = torch.zeros((batch_class_score.shape[0], max_prop_num, batch_box_encoding.shape[2]), device=dev)
    score = torch.zeros((batch_class_score.shape[0], max_prop_num), device=dev)
    box_num_perClass = torch.zeros((batch_class_score.shape[0], 1), device=dev)

    for i in range(batch_class_score.shape[0]):
        _box, _score, box_num_perClass[i][0] = _get_box_score(
            node, batch_class_score[i], batch_box_encoding[i], score_thresh)
        box[i][:_box.shape[0], :], score[i][:_score.shape[0]] = _box, _score

    return box, score, box_num_perClass


@op_register(OpType.Proposal)
def Proposal(self, *args):
    class config(object):
        def __init__(self, scales, ratios, backbone_strides, anchor_stride, std_div, pre_nms_limit):
            self.RPN_ANCHOR_SCALES = scales
            self.RPN_ANCHOR_RATIOS = ratios
            self.BACKBONE_STRIDES = backbone_strides
            self.RPN_ANCHOR_STRIDE = anchor_stride
            self.STD_DIV = std_div
            self.PRE_NMS_LIMIT = pre_nms_limit

    anchor_config = config(scales=[32, 64, 128, 256, 512],
                           ratios=[0.5, 1, 2],
                           backbone_strides=[4, 8, 16, 32, 64],
                           anchor_stride=1,
                           std_div=[10, 10, 5, 5],
                           pre_nms_limit=6000,
                           )

    batch_inp_scores = self.inputs[0].betensor
    batch_inp_bboxes = self.inputs[1].betensor

    box, score = apply_box_deltas(self, anchor_config, batch_inp_scores, batch_inp_bboxes)

    proposal_boxes, proposal_scores, box_num_perClass = get_box_score(self, score, box)

    self.outputs[0].betensor = proposal_scores
    self.outputs[1].betensor = proposal_boxes
    self.outputs[2].betensor = box_num_perClass
    self.outputs[3].betensor = torch.ones(1, 1)

    return [self.outputs[0].betensor, self.outputs[1].betensor, self.outputs[2].betensor, self.outputs[3].betensor]


@quant_register(OpType.Proposal)
def Proposal_quantize(self, *args):
    inp = self.inputs[:]
    out = self.outputs[:]
    # get the height and width of the input image.
    height = self.get_param('height')
    width = self.get_param('width')
    score_thresh = self.get_param('score_threshold')
    STD_DIV = self.get_param('scale_anchor')

    # input scale
    batch_inp_scores_scales,  batch_inp_scores_zp = inp[0].scale, inp[0].zerop
    batch_inp_bboxes_scales,  batch_inp_bboxes_zp = inp[1].scale, inp[1].zerop  # 1/inp[1].scale
    q_bits_activation = self.attrs["q_bits_activation"]
    q_mode_activation = self.attrs["q_mode_activation"]

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

    self.params['height'] = int(height * stat_coords_scale)
    self.params['width'] = int(width * stat_coords_scale)

    wa = self.constants['wa'].betensor
    ha = self.constants['ha'].betensor
    ycenter = self.constants['ycenter'].betensor
    xcenter = self.constants['xcenter'].betensor
    ycenter = linear_quantize_clip(ycenter, stat_coords_scale, stat_coords_zp,
                                   anchor_rmin, anchor_rmax).type(torch.int16)
    xcenter = linear_quantize_clip(xcenter, stat_coords_scale, stat_coords_zp,
                                   anchor_rmin, anchor_rmax).type(torch.int16)
    ha = linear_quantize_clip(ha, stat_coords_scale, stat_coords_zp, anchor_rmin, anchor_rmax).type(torch.int16)
    wa = linear_quantize_clip(wa, stat_coords_scale, stat_coords_zp, anchor_rmin, anchor_rmax).type(torch.int16)

    constants_name = ['ycenter', 'xcenter', 'ha', 'wa']
    for idx, name in enumerate(constants_name):
        self.constants[name] = PyTensor(self.name+name, torch.squeeze(eval(name)
                                                                      ).cpu().numpy().astype(dtype2nptype(bits2dtype(16, is_signed=True))))

    box.qbits = max(16, q_bits_activation)
    box.scale, box.zerop, box.qmin, box.qmax, box.dtype = get_linear_quant_params_from_tensor(
        box, QuantMode.to_symmetric(q_mode_activation), box.qbits, is_signed=True)
    box.qinvariant = False
    coord_zerop = box.zerop
    coord_scale = box.scale
    coord_shift = torch.floor(torch.log2(torch.tensor(coord_scale))).item()  # 2**13
    coord_scale = 2 ** coord_shift

    # coord_scale_q = coord_scale/box_encoding_scale
    def get_lut(in_scale, in_zerop, var, box_scale, box_zerop, lut_in_dtype, lut_size_bits, lut_range_dtype, flag):
        lsteps = 2 ** lut_size_bits
        in_qmin, in_qmax = dtype2range(lut_in_dtype)
        lut_o_qmin, lut_o_qmax = dtype2range(lut_range_dtype)
        lut = linear_dequantize(torch.linspace(in_qmin, in_qmax, steps=lsteps), in_scale, in_zerop)
        lut = lut / var
        if flag:
            lut = torch.exp(lut)
        lut = linear_quantize_clip(lut, box_scale, box_zerop, lut_o_qmin, lut_o_qmax)
        return lut

    var = [10.0, 10.0, 5.0, 5.0]
    # var = self.get_param('variance', optional=True, default_value=_param_default_value['variance'])
    lut_in_dtype = inp[1].dtype
    lut_size_bits = min(inp[1].qbits, int(self.get_attrs('lut_items_in_bits')))
    lut_range_bits = max(self.attrs['q_bits_activation'], 16)
    lut_out_dtype = bits2dtype(lut_range_bits, True)

    ty_lut = get_lut(batch_inp_bboxes_scales, batch_inp_bboxes_zp,
                     var[0], coord_scale, coord_zerop, lut_in_dtype, lut_size_bits, lut_out_dtype, False)
    tx_lut = get_lut(batch_inp_bboxes_scales, batch_inp_bboxes_zp,
                     var[1], coord_scale, coord_zerop, lut_in_dtype, lut_size_bits, lut_out_dtype, False)
    th_lut = get_lut(batch_inp_bboxes_scales, batch_inp_bboxes_zp,
                     var[2], coord_scale, coord_zerop, lut_in_dtype, lut_size_bits, lut_out_dtype, True)
    tw_lut = get_lut(batch_inp_bboxes_scales, batch_inp_bboxes_zp,
                     var[3], coord_scale, coord_zerop, lut_in_dtype, lut_size_bits, lut_out_dtype, True)

    lut_list = [ty_lut, tx_lut, th_lut, tw_lut]
    lut_object_name = {ty_lut: 'ty_lut',
                       tx_lut: 'tx_lut',
                       th_lut: 'th_lut',
                       tw_lut: 'tw_lut'}

    for lut in lut_object_name.keys():
        name = lut_object_name[lut]
        self.constants[name] = PyTensor(
            self.name+name, lut.cpu().numpy().astype(dtype2nptype(bits2dtype(16, is_signed=True))))

    score_q_min, score_q_max = bits2range(inp[0].qbits, False)
    self.params["box_shift_value"] = int(coord_shift)
    self.params["box_shift_type"] = SHIFT_DTYPE
    self.params["score_threshold_value"] = linear_quantize_clip(score_thresh,
                                                                batch_inp_scores_scales, batch_inp_scores_zp, score_q_min, score_q_max).int().item()
    self.params["score_threshold_type"] = bits2dtype(inp[0].qbits, False)

    out_type = [dtype2str(inp[0].dtype), 'int16', 'uint16', 'uint16']
    out_scale = [batch_inp_scores_scales, stat_coords_scale, 1, 1]
    out_zerop = [batch_inp_scores_zp, stat_coords_zp, 0, 0]
    qinvariant_list = [False, False, True, True]
    for idx, out in enumerate(self.outputs):
        dtype = str2dtype(out_type[idx])
        qbits = dtype2bits(dtype)
        out.dtype = dtype
        out.scale = out_scale[idx]
        out.zerop = out_zerop[idx]
        out.qbits = qbits
        out.qinvariant = qinvariant_list[idx]
