# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.ops.eltwise import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch

'''
IR float:
    layer_id=3
    layer_name=score_after_gather
    layer_type=ScatterElements
    layer_bottom=[data,ids,updates]
    layer_bottom_shape=[[16,1717],[16,1717],[16,1717]]
    layer_bottom_type=[int8,uint16,int8]
    layer_top=[score_after_scatter]
    layer_top_shape=[[16,1717]]
    layer_top_type=[int16]
    reduction=ADD
    axis=0
    scale_type=[uint8,uint8,uint8]
    scale_value=[21621,31060,16355]
    shift_type=int8
    shift_value=16
    layer_top_scale=[1.0]
    layer_top_zp=[0]
IR Int:

    layer_id=3
    layer_name=score_after_gather
    layer_type=ScatterElements
    layer_bottom=[data,ids,updates]
    layer_bottom_shape=[[4,4,4],[2,1][2,4]]
    layer_bottom_type=[8bit,uint16/uint8,8bit]
    layer_top=[score_after_scatter]
    layer_top_shape=[[4,4,4]]
    layer_top_type=[8bit]
    reduction = MUL/ADD/NONE
    scale_type = [uint8,uint16,uint16]
    scale_value=[output, data, updata]
    shift_type  = int8
    shift_value = 6


'''


def convert_less_mbit(data, threshold_min, threshold_max):
    left_shift = 0
    while (data > threshold_max) or (data < threshold_min):
        data = data >> 1
        left_shift += 1
    return data, left_shift


@quant_register(OpType.ScatterElements)
def ScatterElements_quantize(self, *args):
    # re-arrange params
    method = self.get_param('reduction', optional=False, default_value='NONE').upper()

    self.params['method'] = method  # copy params
    tmp_inp = self.inputs[1]
    self.replace_input_temporarily(1, self.inputs[2])
    self.replace_input_temporarily(2, tmp_inp)

    eltwise_quantizes(self, *args)

    tmp_inp = self.inputs[1]
    self.replace_input_temporarily(1, self.inputs[2])
    self.replace_input_temporarily(2, tmp_inp)
    self.params.pop('method')

    if method == 'MUL':
        '''
            Scale = [scale2, scale0, scale1]
            Shift = [shift2, shift0, shift1]
        '''
        # rewrite
        q_bits_activation = self.attrs["q_bits_activation"]
        s_mul_out = self.outputs[0].scale
        scale0, scale0_type, shift0, shift0_type = \
            get_scale_approximation_params(s_mul_out, mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)

        s_div_update = 1 / self.inputs[2].scale
        scale1, scale1_type, shift1, shift1_type = \
            get_scale_approximation_params(s_div_update, mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)

        s_div_out = 1 / self.inputs[0].scale
        scale2, scale2_type, shift2, shift2_type = \
            get_scale_approximation_params(s_div_out, mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)

        scales = [int(scale2), int(scale0), int(scale1)]
        shift = [int(shift2), int(shift0), int(shift1)]
        self.params['scale_value'] = scales
        self.params["scale_type"] = [scale2_type, scale0_type, scale1_type]
        self.params['shift_value'] = shift
        self.params["shift_type"] = [shift2_type, shift0_type, shift1_type]
    else:
        # to shrink uint16 dtype into uint8
        scale0, scale1 = self.params['scale_value'][1:]
        shift = self.params['shift_value']
        q_bits_activation = self.attrs["q_bits_activation"]
        while scale0 >= 2 ** q_bits_activation or scale1 >= 2 ** q_bits_activation:
            scale0 = round(scale0 / 2)
            scale1 = round(scale1 / 2)
            shift -= 1
        self.params['scale_value'] = [self.params['scale_value'][0], int(scale0), int(scale1)]
        self.params["scale_type"] = [self.params["scale_type"][0], bits2dtype(q_bits_activation, False),
                                     bits2dtype(q_bits_activation, False)]
        self.params["shift_value"] = shift


@op_register(OpType.ScatterElements)
def ScatterElements(self, *args):
    data = self.inputs[0].betensor  # .reshape(list(self.inputs[0].ir_shape))  # data
    indices = self.inputs[1].betensor.to(torch.long)  # .reshape(list(self.inputs[1].ir_shape))  # indx
    updates = self.inputs[2].betensor  # .reshape(list(self.inputs[2].ir_shape))  # update
    out = self.outputs[0]
    reduce_method = self.get_param('reduction', optional=False, default_value='NONE').upper()  # NONE/ADD/MUL
    if reduce_method not in ['NONE', 'ADD', 'MUL']:
        OPT_ERROR('Scatter_Elements dont support method:%s' % reduce_method)

    axis = self.get_param('axis', optional=False, default_value=0)  # [-s, s-1]
    if not -len(self.inputs[0].ir_shape) <= axis <= len(self.inputs[0].ir_shape)-1:
        OPT_ERROR('Scatter_Elements dont support axis:%s' % axis)

    # to adjust negative index
    max_idx = self.inputs[0].ir_shape[axis]
    indices = torch.where(indices < 0, indices + max_idx, indices)

    if len(self.placeholders) < 1:
        ph0 = PyTensor(self.name+"/tmp_s", torch.tensor(0.).cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
        self.placeholders.append(ph0)

    if self.quantized:
        scale, scale0, scale1 = self.params["scale_value"]
        shift = self.params['shift_value']

        if reduce_method == 'MUL':
            shift, shift0, shift1 = self.params['shift_value']

        inner_min = -2 ** (self.inputs[0].qbits + self.inputs[2].qbits-1)
        inner_max = 2 ** (self.inputs[0].qbits + self.inputs[2].qbits-1) - 1

        if reduce_method == 'ADD':
            data = linear_requantize(data + self.inputs[0].zerop, scale0, 0, 0, inner_min, inner_max)
            updates = linear_requantize(updates + self.inputs[2].zerop, scale1, 0, 0, inner_min, inner_max)
            output = data.clone().scatter_(axis, indices, updates, reduce='add')

        elif reduce_method == 'MUL':
            data = linear_requantize(data + self.inputs[0].zerop, scale0, shift0, 0, inner_min, inner_max)
            #  see bottom for implement details, this code is only for forward speed
            output = data.clone().scatter_(
                axis, indices, (updates + self.inputs[2].zerop)*scale1*0.5**shift1, reduce='multiply')

        elif reduce_method == 'NONE':
            data = linear_requantize(data + self.inputs[0].zerop, scale0, 0, 0, inner_min, inner_max)
            updates = linear_requantize(updates + self.inputs[2].zerop, scale1, 0, 0, inner_min, inner_max)
            output = data.clone().scatter_(axis, indices, updates)

        output = linear_requantize(output, scale, shift, out.zerop, out.qmin, out.qmax)

    else:
        if reduce_method == 'ADD':
            output = data.clone().scatter_(axis, indices, updates, reduce='add')
        elif reduce_method == 'MUL':
            output = data.clone().scatter_(axis, indices, updates, reduce='multiply')
        else:
            output = data.clone().scatter_(axis, indices, updates)  # NONE dont need param 'reduce'

    out.betensor = output
    return out.betensor


'''

def qt_mul(dd, ii, ss, axis, scale2,  scale0, scale1, shift2, shift0, shift1, zp_out=0):
    inner_min, inner_max = -2 ** (qbit + qbit - 1), 2 ** (qbit + qbit - 1) - 1  # 8bit: +-32767,

    dd = torch.clamp(torch.round(dd * scale0*0.5**shift0),inner_min, inner_max)
    rr = dd.clone()

    if len(ii.shape) == 1:
        for i in range(ii.shape[0]):
            tmp_data, ex_shift = convert_less_mbit(rr[ii[i]] * ss[i], inner_min, inner_max)
            rr[ii[i]] = torch.clamp(tmp_data*scale1*0.5**(shift1-ex_shift), inner_min, inner_max)

    if len(ii.shape) == 2:
        for i in range(ii.shape[0]):
            for j in range(ii.shape[1]):
                if axis == 0:
                    tmp_data, ex_shift = convert_less_mbit(rr[ii[i][j]][j] * ss[i][j],inner_min, inner_max)
                    rr[ii[i][j]][j] = torch.clamp(tmp_data*scale1*0.5**(shift1-ex_shift),inner_min, inner_max)

                if axis == 1:
                    tmp_data, ex_shift = convert_less_mbit(rr[i][ii[i][j]] * ss[i][j],inner_min, inner_max)
                    rr[i][ii[i][j]] = torch.clamp(tmp_data*scale1*0.5**(shift1-ex_shift),inner_min, inner_max)
   ...



    rr = torch.clamp(torch.round(rr * scale2 * 0.5 ** shift2)-zp_out, qmin, qmax)

'''
