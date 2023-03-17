# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.ops.eltwise import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
import torch

'''
IR float:
    layer_id=3
    layer_name=ScatterND_0
    layer_type=ScatterND
    layer_bottom=[Placeholder_0,Placeholder_1,Placeholder_2]
    layer_bottom_shape=[[4,4,10,10],[2,1],[2,4,10,10]]
    layer_bottom_type=[float32,int32,float32]
    layer_top=[ScatterND_0]
    layer_top_shape=[[4,4,10,10]]
    layer_top_type=[float32]
    reduction=NONE

IR Int:

    layer_id=3
    layer_name=score_after_gather
    layer_type=ScatterND
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


@quant_register(OpType.ScatterND)
def ScatterND_quantize(self, *args):
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
            Scale = [1, scale0, scale1]
            Shift = [shift, 0, shift1]
        '''
        # rewrite
        q_bits_activation = self.attrs["q_bits_activation"]
        s_data2out = self.outputs[0].scale / self.inputs[0].scale
        scale0, scale0_type, shift0, shift0_type = \
            get_scale_approximation_params(s_data2out, mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)

        s_update2out = 1 / self.inputs[2].scale
        scale1, scale1_type, shift1, shift1_type = \
            get_scale_approximation_params(s_update2out, mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)

        scales = [1, int(scale0), int(scale1)]
        shift = [int(shift0), 0, int(shift1)]
        self.params['scale_value'] = scales
        self.params["scale_type"] = [scale0_type, scale0_type, scale1_type]
        self.params['shift_value'] = shift
        self.params["shift_type"] = [shift0_type, shift0_type, shift1_type]
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


@op_register(OpType.ScatterND)
def ScatterND(self, *args):
    data = self.inputs[0].betensor
    indices = self.inputs[1].betensor.to(torch.long)
    updates = self.inputs[2].betensor
    out = self.outputs[0]
    reduce_method = self.get_param('reduction', optional=False, default_value='NONE').upper()  # NONE/ADD/MUL
    qbits = self.outputs[0].qbits
    if self.quantized:
        scale, scale0, scale1 = self.params["scale_value"]
        inner_min, inner_max = -2 ** 31, 2 ** 31 - 1
        if reduce_method == 'MUL':
            shift, _, shift_ = self.params['shift_value']
        else:
            shift = self.params['shift_value']
        data = (data + self.inputs[0].zerop) * scale0
        updates = (updates + self.inputs[2].zerop) * scale1

    output = torch.clone(data)
    c = [torch.arange(s) for s in indices.shape[:-1]]
    idxs = torch.cartesian_prod(*c) if len(indices.shape) > 1 else [()]

    if len(self.placeholders) < 1:
        ph0 = PyTensor(self.name+"/tmp_s", torch.tensor(0.).cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
        self.placeholders.append(ph0)

    #  scatternd allow add/mul index duplicate
    if reduce_method == 'ADD':
        for idx in idxs:
            if idx != ():
                idx = list(idx) if len(idx.shape) >= 1 else int(idx)
            idxx = list(indices[idx])
            output[idxx] += updates[idx]
            if self.quantized:
                output[idxx] = torch.clamp(output[idxx], inner_min, inner_max)
    if reduce_method == 'MUL':
        if self.quantized:
            for idx in idxs:
                if idx != ():
                    idx = list(idx) if len(idx.shape) >= 1 else int(idx)
                idxx = list(indices[idx])
                if qbits <= 8:
                    output[idxx] = linear_requantize(output[idxx] * updates[idx], 1, shift_, 0, inner_min, inner_max)
                else:  # rounding at 16+bit is much slower than shift
                    output[idxx] = linear_requantize_floor(
                        output[idxx] * updates[idx], 1, shift_, 0, inner_min, inner_max)
        else:
            for idx in idxs:
                if idx != ():
                    idx = list(idx) if len(idx.shape) >= 1 else int(idx)
                idxx = list(indices[idx])
                output[idxx] = output[idxx] * updates[idx]
    if reduce_method == 'NONE':
        for idx in idxs:
            if idx != ():
                idx = list(idx) if len(idx.shape) >= 1 else int(idx)
            idxx = list(indices[idx])
            output[idxx] = updates[idx]
    if self.quantized:
        if qbits <= 8:
            output = linear_requantize(output, scale, shift, out.zerop, out.qmin, out.qmax)
        else:  # rounding at 16+bit is much slower than shift
            output = linear_requantize_floor(output, scale, shift, out.zerop, out.qmin, out.qmax)
    out.betensor = output
    return out.betensor
