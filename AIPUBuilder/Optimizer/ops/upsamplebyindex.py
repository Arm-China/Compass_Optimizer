# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *


@op_register(OpType.UpsampleByIndex)
def upsamplebyindex(self, *args):
    values = self.inputs[0].betensor
    argmax = self.inputs[1].betensor.long()

    current_batch = values.shape[0]
    _, bottom_height, bottom_width, bottom_channels = values.shape
    _, top_height, top_width, top_channels = self.outputs[0].ir_shape

    if bottom_channels != top_channels:
        OPT_ERROR('layer_id=%s, type=%s, bottom channels and top channels are not same, please check!' % (
            self.attrs['layer_id'], str(self.type)))

    flatten_dim = self.get_param('flatten_dim')
    # currently caffe/tensorflow/onnx are not using storage_order
    # So leave this parameter out for now, but keep it for later expansion
    storage_order = self.get_param(
        'storage_order', optional=True, default_value=0)

    total_dims = current_batch * bottom_channels * bottom_height * bottom_width
    chw_dims = bottom_channels * bottom_height * bottom_width
    top_hwc_dims = top_height * top_width * top_channels
    top_hw_dims = top_height * top_width

    top_data = torch.zeros(
        (current_batch, top_height, top_width, top_channels), device=values.device)

    if flatten_dim == 'HW':
        top_data = nhwc2nchw(top_data)
        argmax = nhwc2nchw(argmax)
        values = nhwc2nchw(values)
        for batch in range(current_batch):
            for channel in range(bottom_channels):
                argmax[batch, channel, :, :] += (batch * top_hwc_dims + channel * top_hw_dims)

    elif flatten_dim == 'HWC':
        for batch in range(current_batch):
            argmax[batch, :, :, :] += batch * top_hwc_dims
    elif flatten_dim == 'NHWC':
        pass
    elif flatten_dim == 'NCHW':
        top_data = nhwc2nchw(top_data)
        argmax = nhwc2nchw(argmax)
        values = nhwc2nchw(values)
    else:
        OPT_FATAL("unsupported flatten_dim: %s for OpType(%s) in node:%s" %
                  (flatten_dim, self.type, self.name))

    values_flatten = torch.flatten(values).cpu().numpy()
    top_data_flatten = torch.flatten(top_data).cpu().numpy()
    argmax_flatten = torch.flatten(argmax).long().cpu().numpy()

    length = top_data_flatten.shape[0]
    valid_mask = argmax_flatten < length
    argmax_flatten = argmax_flatten[valid_mask]
    values_flatten = values_flatten[valid_mask]
    top_data_flatten[argmax_flatten] = values_flatten
    out_data = torch.reshape(torch.tensor(top_data_flatten, device=values.device), top_data.shape)

    if flatten_dim in ['NCHW', 'HW']:
        out_data = nchw2nhwc(out_data)

    self.outputs[0].betensor = out_data
    return self.outputs[0].betensor


@quant_register(OpType.UpsampleByIndex)
def upsamplebyindex_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
