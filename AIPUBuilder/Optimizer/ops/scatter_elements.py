# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

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


def get_src_indices_and_update_indices(indices, axis):
    idx_xsection_shape = list(indices.shape[:axis]) + list(indices.shape[axis + 1:])

    def make_slice(arr, axis, i):
        slc = [slice(None)] * arr.ndim
        slc[axis] = i
        return slc

    def unpack(packed):
        unpacked = packed[0]
        for i in range(1, len(packed)):
            unpacked = unpacked, packed[i]
        return unpacked

    def make_indices_for_duplicate(idx):
        final_idx = []
        for i in range(len(idx[0])):
            final_idx.append(tuple(idx_element[i] for idx_element in idx))
        return list(final_idx)

    idx = [[*np.indices(idx_xsection_shape).reshape(indices.ndim - 1, -1),
            indices[tuple(make_slice(indices, axis, i))].reshape(1, -1)[0]] for i in range(indices.shape[axis])]

    idx = list(np.concatenate(idx, axis=1))
    idx.insert(axis, idx.pop())

    updates_idx = list(idx)
    updates_idx.pop(axis)
    updates_idx.insert(
        axis, np.repeat(np.arange(indices.shape[axis]), np.prod(idx_xsection_shape))
    )
    idx, updates_idx = make_indices_for_duplicate(idx), make_indices_for_duplicate(
        updates_idx
    )
    return idx, updates_idx


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
        # rewrite
        q_bits_activation = self.attrs["q_bits_activation"]
        axis = self.params['axis']
        max_multiply_count = list(self.inputs[1].ir_shape)[axis]

        scale0, scale0_type, shift0, shift0_type = \
            get_scale_approximation_params(self.outputs[0].scale / self.inputs[0].scale, mult_bits=14,
                                           force_shift_positive=self.force_shift_positive)

        scale1, scale1_type, shift1, shift1_type = \
            get_scale_approximation_params(1 / self.inputs[2].scale, mult_bits=14,
                                           force_shift_positive=self.force_shift_positive)
        scales = [int(scale0), int(scale1)]
        shift = [int(shift0), int(shift1)]
        self.params['scale_value'] = scales
        self.params["scale_type"] = [scale0_type, scale1_type]
        self.params['shift_value'] = shift
        self.params["shift_type"] = [shift0_type, shift1_type]
        ####################################version 2######################################
        # scales = [int(scale0)]
        # shifts = [int(shift0)]
        # step_scales = int(1)
        # step_shifts = int(0)
        # for i in range(1, max_multiply_count+1):
        #     step_scales *= scale1
        #     step_scales, left_shift = convert_less_mbit(step_scales, -2**15, (2**15)-1)
        #     scales.append(int(step_scales))
        #     step_shifts = step_shifts + shift1 - left_shift
        #     shifts.append(int(step_shifts))
        # do_scale = torch.tensor(scales, device = self.inputs[0].betensor.device)
        # do_shift = torch.tensor(shifts, device=self.inputs[0].betensor.device)
        # self.set_ir_field('scale', do_scale, scale0_type)
        # self.set_ir_field('shift', do_shift, shift0_type)
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
    indices = self.inputs[1].betensor.to(torch.long)
    out = self.outputs[0]
    reduce_method = self.get_param('reduction', optional=False, default_value='NONE').upper()  # NONE/ADD/MUL
    if reduce_method not in ['NONE', 'ADD', 'MUL', 'MIN', 'MAX']:
        OPT_ERROR('Scatter_Elements dont support method:%s' % reduce_method)

    axis = self.get_param('axis', optional=False, default_value=0)  # [-s, s-1]
    if not -len(self.inputs[0].ir_shape) <= axis <= len(self.inputs[0].ir_shape)-1:
        OPT_ERROR('Scatter_Elements dont support axis:%s' % axis)

    # to adjust negative index
    max_idx = self.inputs[0].ir_shape[axis]
    indices = torch.where(indices < 0, indices + max_idx, indices)
    indices = torch.clamp(indices, 0, max_idx - 1)

    if len(self.placeholders) < 1:
        ph0 = PyTensor(self.name+"/tmp_s", torch.tensor(0.), dtype=Dtype.FP32)
        self.placeholders.append(ph0)

    if self.quantized:
        data = self.inputs[0].betensor.long()
        updates = self.inputs[2].betensor.long()
        if reduce_method == 'MUL':
            scale = self.params['scale_value']
            shift = self.params['shift_value']
            output_signed = is_signed(self.outputs[0].dtype)
            qmin, qmax = bits2range(16, output_signed)
            data = linear_requantize(data + self.inputs[0].zerop, scale[0], shift[0], 0, qmin, qmax).int()
            updates += self.inputs[2].zerop
            idx, updates_idx = get_src_indices_and_update_indices(indices.cpu().numpy(), axis)

            scattered = data.clone().int()
            idx_dict = {}
            for iter, idx_set in enumerate(idx):
                scattered[idx_set] *= (updates[updates_idx[iter]].int())
                scattered[idx_set], left_shift = convert_less_mbit(scattered[idx_set], qmin, qmax)
                scattered[idx_set] = (scattered[idx_set] * scale[1]) >> (shift[1] - left_shift)
            #     if idx_set not in idx_dict.keys():
            #         idx_dict[idx_set] = shift[1] - left_shift
            #     else:
            #         idx_dict[idx_set] += (shift[1] - left_shift)
            # for key in idx_dict.keys():
            #     scattered[key] = linear_requantize(scattered[key], 1, idx_dict[key], 0, qmin, qmax)#(scattered[idx_set] * scale[1]) >> (shift[1] - left_shift)
            output = torch.clamp(scattered, self.outputs[0].qmin, self.outputs[0].qmax)

        else:
            shift, shift0, shift1 = self.params['shift_value'], 0, 0
            scale, scale0, scale1 = self.params["scale_value"]
            if is_signed(self.outputs[0].dtype):
                inner_min = -2 ** (self.inputs[0].qbits + self.inputs[2].qbits-1)
                inner_max = 2 ** (self.inputs[0].qbits + self.inputs[2].qbits-1) - 1
            else:
                inner_min = 0
                inner_max = 2 ** (self.inputs[0].qbits + self.inputs[2].qbits) - 1

            data = linear_requantize(data + self.inputs[0].zerop, scale0, shift0, 0, inner_min, inner_max)
            updates = linear_requantize(updates + self.inputs[2].zerop, scale1, shift1, 0, inner_min, inner_max)
            data = data.to(updates.dtype)

            if reduce_method == 'ADD':
                output = data.clone().scatter_(axis, indices, updates, reduce='add')

            elif reduce_method == 'NONE':
                output = data.clone().scatter_(axis, indices, updates)

            elif reduce_method in ['MIN', 'MAX']:
                output = data.clone().scatter_reduce(axis, indices, updates, reduce='a'+reduce_method.lower(), include_self=True)

            output = linear_requantize(output, scale, shift, out.zerop, out.qmin, out.qmax)

    else:
        data = self.inputs[0].betensor.float()
        updates = self.inputs[2].betensor.float()
        if reduce_method == 'ADD':
            output = data.clone().scatter_(axis, indices, updates, reduce='add')
        elif reduce_method == 'MUL':
            output = data.clone().scatter_(axis, indices, updates, reduce='multiply')
        elif reduce_method == 'NONE':
            output = data.clone().scatter_(axis, indices, updates)  # NONE dont need param 'reduce'
        elif reduce_method in ['MIN', 'MAX']:
            output = data.clone().scatter_reduce(axis, indices, updates, reduce='a'+reduce_method.lower(), include_self=True)

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
