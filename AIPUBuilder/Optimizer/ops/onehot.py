# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *


@op_register(OpType.OneHot)
def onehot(self, *args):
    '''
        axis=-1
        depth=8
        values=[0.0,1.0]
        on_value=255/127 [decrepated]
        off_value=0/-128[decrepated]
    '''

    inp = self.inputs[0].betensor.long()
    out = self.outputs[0]
    off_value = self.params['values'][0]
    on_value = self.params['values'][1]
    axis = self.get_param('axis')
    new_axis = axis if axis >= 0 else len(inp.shape)
    # if axis exceeds length of input shape, do not need transpose
    if axis > len(inp.shape) - 1:
        axis = new_axis = -1
    # pytorch one_hot only support last dim indice,so we need transpose inp
    # according ir definition:
    # axis =-1, shape=batch,feature,num_class
    # axis = 0, shape=num_class,batch,feature
    # axis = 1, shape=batch,num_class,feature
    # indice is negatibe and out of depth, 0 replace
    classes_number = int(self.params['depth'])

    trans_inp = torch.transpose(inp.unsqueeze(axis), axis, -1)
    legal_data_flag = (trans_inp >= 0) * (trans_inp < classes_number)
    trans_inp = trans_inp*legal_data_flag
    trans_y = torch.nn.functional.one_hot(trans_inp, classes_number)
    trans_y[..., 0] = trans_y[..., 0]*legal_data_flag
    y = torch.transpose(trans_y.squeeze(-2), new_axis, -1)
    out.betensor = y*(on_value-off_value) + off_value
    return out.betensor


@quant_register(OpType.OneHot)
def onehot_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    off_value_ = self.params['values'][0]
    on_value_ = self.params['values'][1]
    quan_off_value, quan_on_value = self.params['values'][0], self.params['values'][1]
    # torch.max(abs(torch.Tensor([quan_on_value, quan_off_value])))
    on_off_max = max(abs(quan_on_value), abs(quan_off_value))
    obits = self.attrs['q_bits_activation']

    if self.params['values'][0] < 0 or self.force_dtype_int:
        osigned = True
        o_min = -(2**(obits-1))
        o_max = (2**(obits-1)-1)
        out_scale = (2**(obits-1)-1) / on_off_max
        quan_on_value = max(min(int(round(on_value_ * out_scale)), o_max), o_min)
        quan_off_value = max(min(int(round(off_value_ * out_scale)), o_max), o_min)
    else:
        osigned = False
        o_min = 0
        o_max = (2**(obits)-1)
        out_scale = (2**(obits)-1) / on_off_max
        quan_on_value = max(min(int(round(on_value_ * out_scale)), o_max), o_min)
        quan_off_value = max(min(int(round(off_value_ * out_scale)), o_max), o_min)

    # self.params['on_value']=quan_on_value
    # self.params['off_value']=quan_off_value
    self.params['values'] = [quan_off_value, quan_on_value]
    self.params['depth'] = int(self.params['depth'])
    out.scale = out_scale
    out.zerop = 0
    out.qbits = obits
    out.dtype = bits2dtype(obits, is_signed=osigned, use_float=False)
    out.qinvariant = inp.qinvariant
