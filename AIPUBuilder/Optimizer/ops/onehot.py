# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

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
    invalid_mask = torch.bitwise_or(trans_inp < 0, trans_inp >= classes_number)
    trans_inp[invalid_mask] = 0
    # legal_data_flag = (trans_inp >= 0) * (trans_inp < classes_number)
    # trans_inp = trans_inp*legal_data_flag
    trans_y = torch.nn.functional.one_hot(trans_inp, classes_number)
    trans_y[invalid_mask, :] = 0
    # trans_y[..., 0] = trans_y[..., 0]*legal_data_flag
    y = torch.transpose(trans_y.squeeze(-2), new_axis, -1)
    out.betensor = y*(on_value-off_value) + off_value
    if self.quantized:
        out.betensor = torch.clamp(out.betensor, out.qmin, out.qmax).int()
    return out.betensor


@quant_register(OpType.OneHot)
def onehot_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    q_mode_activation = self.attrs["q_mode_activation"]
    off_value_ = self.params['values'][0]
    on_value_ = self.params['values'][1]
    min_value = min(off_value_, on_value_)
    max_value = max(off_value_, on_value_)
    obits, out_signed = range2bits(min_value, max_value)
    if inp.qinvariant:
        if (type(off_value_) == int and type(on_value_) == int) \
                or ((off_value_ - int(off_value_) < OPT_EPSILON) and (on_value_ - int(on_value_) < OPT_EPSILON)):
            out.qinvariant = True
            out.scale = 1
            out.zerop = 0
            out.qbits = max(obits, 8)
            out.dtype = bits2dtype(out.qbits, is_signed=out_signed)
            out.qmin, out.qmax = dtype2range(out.dtype)
        else:
            out.min = min_value
            out.max = max_value
            out.qbits = self.attrs['q_bits_activation']
            out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
                out, q_mode_activation, out.qbits, is_signed=out_signed)
            out.qinvariant = False
            quan_on_value = linear_requantize(on_value_, out.scale, 0, out.zerop, out.qmin, out.qmax).int().item()
            quan_off_value = linear_requantize(off_value_, out.scale, 0, out.zerop, out.qmin, out.qmax).int().item()
            self.params['values'] = [quan_off_value, quan_on_value]
    else:
        OPT_WARN(
            'currently only support input qinvariant is True. but currently input qinvariant is False.layer_id=%s, layer_name=%s' % (
                self.attrs['layer_id'], self.name),
            workflow_name='forward', op_name=str(self.type))
        out.qinvariant = inp.qinvariant
        out.scale = inp.scale
        out.zerop = inp.zerop
        out.qbits = inp.qbits
        out.dtype = inp.dtype
        out.qmin = inp.qmin
        out.qmax = inp.qmax
