# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.ops.pad import pad
from AIPUBuilder.Optimizer.ops.multibox_transform_Loc import calculate_box_quantization, apply_box_deltas, apply_box_deltas_q
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import OPT_DEBUG, OPT_INFO, OPT_ERROR

# IR
# layer_name=mrcnn_detecion/decodebox
# layer_type=BoundingBox
# layer_bottom=[post_nms1_proposal_bbox_tensor,mrcnn_detecion/reshape_max_deltas_output]
# layer_bottom_shape=[[1,1000,4],[1,1000,4]]
# layer_bottom_type=[float32,float32]
# layer_top=[mrcnn_detecion/decodebox_output]
# layer_top_shape=[[1,1000,4]]
# layer_top_type=[float32]

register_optype('BoundingBox')


@op_register(OpType.BoundingBox)
def boundingbox(self, *args):
    proposal_box = self.inputs[0].betensor + (torch_tensor(0, device=self.inputs[0].device)
                                              if not self.quantized else self.inputs[0].zerop)
    box_delta = self.inputs[1].betensor

    if not self.quantized:
        refined_box = apply_box_deltas(self, proposal_box, box_delta)
    else:
        # x,y (box_delta[0:2]) need convert to symetric,dh,dw as lut index, zerop is aborbed to lut
        box_delta[..., 0:2] = box_delta[..., 0:2] + self.inputs[1].zerop
        refined_box = apply_box_deltas_q(self, proposal_box, box_delta, self.inputs[0].dtype, self.inputs[1].dtype)
        '''
        #x,y (box_delta[0:2]) need convert to symetric,dh,dw as lut index, zerop is aborbed to lut
        box_delta[...,0:2] = box_delta[...,0:2]+torch.tensor(self.inputs[1].zerop)
        refined_box = apply_box_deltas_q(self, proposal_box, box_delta, self.inputs[0].dtype, self.inputs[1].dtype)
        '''
    self.outputs[0].betensor = refined_box
    return refined_box


@quant_register(OpType.BoundingBox)
def boundingbox_quantize(self, *args):
    # batc_rois, batch_probs, batch_deltas
    inp = self.inputs[:]
    out = self.outputs[:]

    # ################################get initial params###########################################
    STD_DIV = self.get_param('std_div', optional=True, default_value=[10, 10, 5, 5])

    calculate_box_quantization(self, inp[0], inp[1], STD_DIV)
    '''
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
    #calculate_box_quantization(self, inp[0], inp[1], STD_DIV)
    '''
    # set dtpye and qbits
    out[0].scale, out[0].zerop, out[0].dtype, out[0].qbits = 32767., 0, Dtype.INT16, 16     # box
    out[0].qinvariant = inp[0].qinvariant
    # out[1] and out[2] will be deleted in IR
    for idx in range(1, len(self.outputs)):
        out[idx].scale = 1.
        out[idx].zerop = 0
        out[idx].dtype = Dtype.UINT16
        out[idx].qbits = 16
        out[idx].qinvariant = True
