# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
from AIPUBuilder.Optimizer.framework import *
import math


@op_register(OpType.PostNMS1)
def PostNms1(self, *args):
    out = self.outputs[:]
    # process parameters
    proposal_cnt = self.get_param('proposal_cnt')
    # get the height and width of the input image.
    height = self.get_param('image_height')
    width = self.get_param('image_width')

    # get bottom node
    bottom = []
    for i, inp in enumerate(self.inputs):
        bottom.append(inp.betensor)

    [proposal_boxes, keep, nms_box_num_perClass] = bottom[:]

    max_prop_num = proposal_boxes.shape[1]

    keep = keep.int()
    nms_box_num_perClass = nms_box_num_perClass.int()
    max_nms_class_num = nms_box_num_perClass.shape[1]

    batch_num = proposal_boxes.shape[0]

    out_proposal_box = torch.zeros((batch_num, proposal_cnt, 4), device=proposal_boxes.device)
    nor_boxes = torch.zeros((batch_num, proposal_cnt, 4), device=proposal_boxes.device)
    for batch_idx in range(batch_num):
        cur_box_idx = 0
        box_idx_cur_class = 0
        for class_idx in range(max_nms_class_num):
            for box_perclass_idx in range(nms_box_num_perClass[batch_idx, class_idx]):
                if cur_box_idx >= proposal_cnt:  # output the first proposal_cnt boxes
                    break
                box_idx_classes = box_idx_cur_class + keep[batch_idx, cur_box_idx]
                if box_idx_classes >= max_prop_num:
                    break
                out_proposal_box[batch_idx, cur_box_idx] = proposal_boxes[batch_idx, box_idx_classes, :]
                cur_box_idx += 1

            box_idx_cur_class += nms_box_num_perClass[batch_idx, class_idx]

    max_inh_inw = max(height, width)

    if not self.quantized:
        nor_boxes = out_proposal_box / max_inh_inw
    else:
        nor_box_shift_h = self.params['nor_box_shift_h']
        nor_box_inh_scale_h = self.params['nor_box_scale_h']

        nor_box_shift_w = self.params['nor_box_shift_w']
        nor_box_inh_scale_w = self.params['nor_box_scale_w']
        # nor_boxes = np.array(out_proposal_box, dtype=self.top_type_['int8'][0])
        nor_boxes[..., 0:4:2] = linear_requantize(out_proposal_box[:, :, 0:4:2]+self.inputs[0].zerop,
                                                  nor_box_inh_scale_h, nor_box_shift_h, out[1].zerop, out[1].qmin, out[1].qmax)
        nor_boxes[..., 1:4:2] = linear_requantize(out_proposal_box[:, :, 1:4:2]+self.inputs[0].zerop,
                                                  nor_box_inh_scale_w, nor_box_shift_w, out[1].zerop, out[1].qmin, out[1].qmax)

        out_proposal_box = torch.clamp(out_proposal_box, out[0].qmin, out[0].qmax)

    out[0].betensor = out_proposal_box
    out[1].betensor = nor_boxes
    return [o.betensor for o in self.outputs]


@quant_register(OpType.PostNMS1)
def PostNms1_quantize(self, *args):
    inp = self.inputs[:]
    out = self.outputs[:]
    out[0].dtype = inp[0].dtype
    out[0].qbits = inp[0].qbits
    out[0].qinvariant = inp[0].qinvariant
    proposal_cnt = self.get_param('proposal_cnt')
    # input scale
    box_encoding_scale, box_encoding_zp = inp[0].scale, inp[0].zerop
    nor_box_scale, nor_box_zp = box_encoding_scale, box_encoding_zp
    # get the height and width of the input image.
    height = self.get_param('image_height')
    width = self.get_param('image_width')
    q_mode_activation = self.attrs["q_mode_activation"]
    sign = is_signed(Dtype.UINT16)
    out1 = self.outputs[1]  # normalize box tesnor
    out1.qbits = 16
    out1.dtype = bits2dtype(out1.qbits, is_signed=sign)
    out1.qmin, out1.qmax = dtype2range(out1.dtype)
    sim_input = out[0]
    sim_input.min, sim_input.max = 0, nor_box_scale * height
    nor_box_inh_scale, nor_box_inh_zerop, _, _, _ = get_linear_quant_params_from_tensor(
        sim_input, q_mode_activation, sim_input.qbits, False)
    extra_params = self.get_attrs('extra_params', optional=True, default_value=[0, 10])
    nor_box_shift = extra_params[1]
    sim_input.min, sim_input.max = 0., nor_box_scale * width
    nor_box_inw_scale, nor_box_inw_zerop, _, _, _ = get_linear_quant_params_from_tensor(
        sim_input, q_mode_activation, sim_input.qbits, False)

    out[1].scale, out[1].zerop = nor_box_inh_scale*max(height, width)*nor_box_scale, nor_box_inh_zerop
    nor_box_inh_scale = math.floor(nor_box_inh_scale * (2**nor_box_shift))
    nor_box_inw_scale = math.floor(nor_box_inw_scale * (2**nor_box_shift))

    out[0].scale, out[0].zerop = box_encoding_scale, box_encoding_zp
    out[1].qinvariant = inp[0].qinvariant

    # int IR params
    self.params["proposal_cnt"] = proposal_cnt
    self.params["nor_box_scale_h"] = nor_box_inh_scale
    self.params["nor_box_scale_w"] = nor_box_inw_scale

    self.params["nor_box_shift_h"] = nor_box_shift
    self.params["nor_box_shift_w"] = nor_box_shift
