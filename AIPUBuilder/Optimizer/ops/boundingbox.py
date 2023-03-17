# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.ops.pad import pad
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import OPT_DEBUG, OPT_INFO, OPT_ERROR

# IR
# layer_name=mrcnn_detecion/decodebox
# layer_type=DetectionBox
# layer_bottom=[post_nms1_proposal_bbox_tensor,mrcnn_detecion/reshape_max_deltas_output]
# layer_bottom_shape=[[1,1000,4],[1,1000,4]]
# layer_bottom_type=[float32,float32]
# layer_top=[mrcnn_detecion/decodebox_output]
# layer_top_shape=[[1,1000,4]]
# layer_top_type=[float32]


def apply_box_deltas(self, boxes, deltas):
    ycenter_a = (boxes[0, :, 0] + boxes[0, :, 2]) / 2
    xcenter_a = (boxes[0, :, 1] + boxes[0, :, 3]) / 2
    ha = boxes[0, :, 2] - boxes[0, :, 0]
    wa = boxes[0, :, 3] - boxes[0, :, 1]

    std_div = self.get_param('std_div', optional=True, default_value=[10, 10, 5, 5])

    dy = deltas[0, :, 0] / std_div[0]
    dx = deltas[0, :, 1] / std_div[1]
    dh = deltas[0, :, 2] / std_div[2]
    dw = deltas[0, :, 3] / std_div[3]

    # adjust achors size and position
    ycenter = dy * ha + ycenter_a
    xcenter = dx * wa + xcenter_a
    h = torch.exp(dh) * ha
    w = torch.exp(dw) * wa

    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    box = torch.zeros_like(boxes)
    box[-1:] = torch.stack([ymin, xmin, ymax, xmax], dim=1)
    # print("box(after apply delta) before clip:",np.min(box), np.max(box))
    # clip in normalized size[0~1.0]
    #  window: (y1, x1, y2, x2)
    box[0, :, 0][box[0, :, 0] < 0] = 0
    box[0, :, 1][box[0, :, 1] < 0] = 0
    box[0, :, 2][box[0, :, 2] < 0] = 0
    box[0, :, 3][box[0, :, 3] < 0] = 0
    box[0, :, 0][box[0, :, 0] > 1] = 1
    box[0, :, 1][box[0, :, 1] > 1] = 1
    box[0, :, 2][box[0, :, 2] > 1] = 1
    box[0, :, 3][box[0, :, 3] > 1] = 1

    return box


def apply_box_deltas_q(self, boxes, deltas):
    if self.quantized:
        # check input delta box bits
        qbits = dtype2bits(self.inputs[1].dtype)
        if qbits not in [8, 16]:
            OPT_ERROR(f"qbits={qbits} does not support in BoundingBox layer")
        scales = self.params["scale_value"]
        shifts = self.params["shift_value"]

        ycenter = (boxes[0, :, 0] + boxes[0, :, 2]).int() >> 1
        xcenter = (boxes[0, :, 1] + boxes[0, :, 3]).int() >> 1
        h = boxes[0, :, 2] - boxes[0, :, 0]
        w = boxes[0, :, 3] - boxes[0, :, 1]

        dy = deltas[0, :, 0].int()
        dx = deltas[0, :, 1].int()
        dh = deltas[0, :, 2].int()
        dw = deltas[0, :, 3].int()
        scale_txty_out = scales[0]
        shift_txty_out = shifts[0]
        # sat_bits is to avoid lib calculation saturation,here,so it need (qbits*2+16-32)
        # assume scale is 16bits, max bits 32, currently, only support 8bits and 16bits box
        sat_bits = 2*qbits+16-32
        round_shift = 1 << (shift_txty_out-sat_bits-1)
        clip_min, clip_max = bits2range(32, True)

        h_scale = (h * scale_txty_out) >> sat_bits
        w_scale = (w * scale_txty_out) >> sat_bits
        ycenter_q = torch.clip(((dy * h_scale+round_shift) >> (shift_txty_out-sat_bits)) + ycenter, clip_min, clip_max)
        xcenter_q = torch.clip(((dx * w_scale+round_shift) >> (shift_txty_out-sat_bits)) + xcenter, clip_min, clip_max)

        lut = self.constants['lut'].betensor
        scale_thtw_out = scales[1]
        shift_thtw_out = shifts[1]
        lut_offset = 1 << (qbits-1)
        clip_min, clip_max = bits2range((qbits*2), False)
        lut_dh_scale = ((lut[dh.long()+lut_offset] * scale_thtw_out) * 2**(-16)).int()
        lut_dw_scale = ((lut[dw.long()+lut_offset] * scale_thtw_out) * 2**(-16)).int()
        h_half_q = torch.clip((h * lut_dh_scale * 2**(16-shift_thtw_out)), clip_min, clip_max)
        w_half_q = torch.clip((w * lut_dw_scale * 2**(16-shift_thtw_out)), clip_min, clip_max)

        scale_anchor = scales[2]
        shift_anchor = shifts[2]
        round_shift = 0
        if shift_anchor > 1:
            round_shift = 1 << (shift_anchor-1)

        ymin_q32 = (ycenter_q - h_half_q).int() * scale_anchor+round_shift >> shift_anchor
        xmin_q32 = (xcenter_q - w_half_q).int() * scale_anchor+round_shift >> shift_anchor
        ymax_q32 = (ycenter_q + h_half_q).int() * scale_anchor+round_shift >> shift_anchor
        xmax_q32 = (xcenter_q + w_half_q).int() * scale_anchor+round_shift >> shift_anchor

        box_q32 = torch.stack([ymin_q32, xmin_q32, ymax_q32, xmax_q32], dim=1)
        box_q16 = torch.zeros_like(boxes)
        box_q16[-1:] = torch.clip(box_q32, 0, 32767)  # clip to 0~1.0

        return box_q16
    else:
        OPT_ERROR("in detection quantize routine, node should be quantized.")


register_optype('BoundingBox')


@op_register(OpType.BoundingBox)
def boundingbox(self, *args):
    proposal_box = self.inputs[0].betensor + (torch.tensor(
        0) if not self.quantized else torch.tensor(self.inputs[0].zerop))
    box_delta = self.inputs[1].betensor

    if not self.quantized:
        refined_box = apply_box_deltas(self, proposal_box, box_delta)
    else:
        # x,y (box_delta[0:2]) need convert to symetric,dh,dw as lut index, zerop is aborbed to lut
        box_delta[..., 0:2] = box_delta[..., 0:2]+torch.tensor(self.inputs[1].zerop)
        refined_box = apply_box_deltas_q(self, proposal_box, box_delta)
    self.outputs[0].betensor = refined_box
    return refined_box


@quant_register(OpType.BoundingBox)
def boundingbox_quantize(self, *args):
    # batc_rois, batch_probs, batch_deltas
    inp = self.inputs[:]
    out = self.outputs[:]

    # ################################get initial params###########################################
    STD_DIV = self.params['std_div'] if 'std_div' in self.params else [10, 10, 5, 5]

    # ##############################apply deltas part######################################
    # # input scale
    # batch_inp_scores_scales, batch_inp_scores_zp =  inp[0].scale, inp[0].zerop
    batch_inp_rois_scales,   batch_inp_rois_zp = inp[0].scale, inp[0].zerop
    batch_inp_deltas_scales, batch_inp_deltas_zp = inp[1].scale, inp[1].zerop

    scale_txty_out = batch_inp_deltas_scales * STD_DIV[0]
    qbits = inp[1].qbits
    # if qbits == 8:
    # poposal delta variance >> detection delta, so we need limit exp(x), if x too large, will affect acc
    # and x is too large, it is meaningless, so limit max delta which is calculated according to scale_thtw_out
    # we hope scale_thtw_out is at least 10
    max_delta = torch.tensor(3276.0).log().item()*STD_DIV[2]
    half_max = (1 << (qbits-1))-1+batch_inp_deltas_zp
    if half_max > batch_inp_deltas_scales*max_delta:
        batch_inp_deltas_scales = half_max/max_delta
    scale_thtw_out = 32767/self.params['max_exp_thtw'] if 'max_exp_thtw' in self.params else 32767 / \
        torch.tensor(half_max/batch_inp_deltas_scales/STD_DIV[2]).exp()
    # elif qbits == 16:
    #     max_delta = torch.tensor(3276.0).log().item()*STD_DIV[2]
    #     if 32767 > batch_inp_deltas_scales*max_delta:
    #         batch_inp_deltas_scales = 32767/max_delta
    #     scale_thtw_out = 32767/self.params['max_exp_thtw'] if 'max_exp_thtw' in self.params else 32767 / \
    #         torch.tensor(32767/batch_inp_deltas_scales/STD_DIV[2]).exp()

    dev = inp[1].betensor.device
    if inp[1].qbits > 16:
        OPT_WARN(
            self.name + " : input tensor use %d bit quantization, which may cause proposal's lut table very large." % inp[1].qbits)
    max_value = 1 << qbits
    half_max_value = max_value >> 1
    # if qbits == 8:
    float_x = ((torch.arange(0, max_value)-half_max_value)+batch_inp_deltas_zp)/batch_inp_deltas_scales
    float_y = torch.exp(float_x/STD_DIV[2])
    lut_16 = torch.clamp(torch.round(float_y * scale_thtw_out), 0, 32767)

    # for lib requirement's int, in fact, int16 is enough
    lut_32 = lut_16.int()
    self.constants["lut"] = PyTensor(self.name+"/detection_lut", lut_32)
    # elif qbits == 16:
    #     float_x = (torch.arange(0, 65536) - 32768) / batch_inp_deltas_scales  # 258: for -127~127 strict quant mode.
    #     float_y = torch.exp(float_x/STD_DIV[2])
    #     lut_16 = torch.clamp(torch.round(float_y * scale_thtw_out), 0, 32767).short()
    #     # for lib requirement's int, in fact, int16 is enough
    #     lut_32 = lut_16.int()
    #     self.constants["lut"] = PyTensor(self.name+"/detection_lut", lut_32)
    # scale for detla (x,y,w,h)
    xy_scale, xy_scale_type, xy_shift, xy_shift_type = \
        get_scale_approximation_params(1/scale_txty_out, mult_bits=16, force_shift_positive=self.force_shift_positive)
    hw_scale, hw_scale_tyep, hw_shift, hw_shift_type = \
        get_scale_approximation_params(1/scale_thtw_out, mult_bits=16, force_shift_positive=self.force_shift_positive)
    hw_shift = hw_shift + 1  # for half calculate
    # convert input box scale(batch_inp_rois_scales) to output box (32767)
    anchor_scale, anchor_scale_type, anchor_shift, anchor_shift_type = \
        get_scale_approximation_params(32767/batch_inp_rois_scales, mult_bits=15,
                                       force_shift_positive=self.force_shift_positive)

    self.params["scale_value"] = [int(xy_scale), int(hw_scale), int(anchor_scale)]
    self.params["scale_type"] = [xy_scale_type, hw_scale_tyep, anchor_scale_type]
    self.params["shift_value"] = [int(xy_shift), int(hw_shift), int(anchor_shift)]
    self.params["shift_type"] = [xy_shift_type, hw_shift_type, anchor_shift_type]

    # set dtpye and qbits
    out[0].scale, out[0].zerop, out[0].dtype, out[0].qbits = 32767, 0, Dtype.INT16, 16     # box
    out[0].qinvariant = inp[0].qinvariant
    out[1].scale, out[1].zerop, out[1].dtype, out[1].qbits = 1, 0, Dtype.UINT16, 16   # score
    out[1].qinvariant = True
    out[2].scale, out[2].zerop, out[2].dtype, out[2].qbits = 1, 0, Dtype.UINT16, 16
    out[2].qinvariant = True
