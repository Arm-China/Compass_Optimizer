# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch

###########################################################
# layer_id=2
# layer_name=SegmentSum
# layer_type=SegmentReduce
# layer_bottom=[Placeholder,SegmentSum/segment_ids]
# layer_bottom_shape=[[5,1,2,3],[5]]
# layer_bottom_type=[float32,int32]
# layer_top=[SegmentSum]
# layer_top_shape=[[5,1,2,3]]
# layer_top_type=[float32]
# method=SUM

#inp0 = [0, 1, 3, 5, 6, 10, 12, 17, 20]
#index= [0, 0, 0, 2, 4, 7,   8,  9, 10]
#output=[4, 0, 5, 0, 6, 0, 0, 10, 12, 17, 20]
###########################################################


@quant_register(OpType.SegmentReduce)
def segmentReduce_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]
    method = self.get_param('method')

    inp = self.inputs[0]
    out = self.outputs[0]
    out.qbits = q_bits_activation
    out_sign = is_signed(inp.dtype)
    dev = inp.betensor.device
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, out_sign)
    if method == 'SUM':
        local_rescale = out.scale / inp.scale
        scale_bits = max(16, q_bits_activation)
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(local_rescale, mult_bits=scale_bits,
                                           force_shift_positive=self.force_shift_positive)
        self.params["shift_value"] = int(do_shift)
        self.params["shift_type"] = do_shift_type
        self.params["scale_value"] = int(do_scale)
        self.params["scale_type"] = do_scale_type
        out.qinvariant = False
    else:
        OPT_FATAL("unsupported method: %s for segmentReduce in node:%s" % (method, self.name))


@op_register(OpType.SegmentReduce)
def segmentreduce(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    method = self.get_param('method').upper()
    if method not in {"SUM"}:
        OPT_FATAL("unsupported method: %s for segmentReduce in node:%s" % (method, self.name))
    input_data = inp0.betensor
    segment_index = inp1.betensor
    if self.quantized:
        input_data = (input_data + inp0.zerop).long()
        segment_index += inp1.zerop
    segment_index = segment_index.reshape(-1,).int()
    index_size = segment_index.numel()
    # segment_index first dim must equal to input_data first dim currently
    # so currently set axis is 0, read from floatIR in furture
    axis = 0
    input_data_dim = input_data.shape
    data_dim_axis = input_data.shape[axis]
    if index_size > data_dim_axis:
        segment_index = segment_index[:data_dim_axis]
        OPT_WARN('layer_id=%s, type=%s, index size is more than data %s-th dimension, so intercept the preceding dimensions of index'
                 % (self.attrs['layer_id'], str(self.type), str(axis)))
    elif index_size < data_dim_axis:
        segment_index = torch.nn.functional.pad(
            segment_index, (0, data_dim_axis-index_size), value=segment_index[-1].item())
        OPT_WARN('layer_id=%s, type=%s, index size is less than data %s-th dimension, so pad the dimensions of index'
                 % (self.attrs['layer_id'], str(self.type), str(axis)))

    method_d = {
        "SUM": torch.sum,
    }
    op = method_d[method]

    index_range = (segment_index[-1] + 1).item()
    zero_dim = list(input_data_dim)
    zero_dim[axis] = 1
    zero_value = 0
    output = None
    if method == 'SUM':
        for j in range(index_range):
            if j in segment_index:
                j_index = torch.eq(segment_index, j)
                j_index_range = torch.nonzero(j_index)
                tmp_data = torch.index_select(input_data, axis, torch.squeeze(j_index_range))
                tmp_data = op(tmp_data, axis, keepdim=True)
            else:
                tmp_data = torch.full((zero_dim), zero_value, device=inp0.betensor.device)
            output = tmp_data if output == None else torch.cat((output, tmp_data), dim=axis)

    if self.quantized:
        do_shift = self.params["shift_value"]
        do_scale = self.params["scale_value"]
        output = linear_requantize(output, do_scale, do_shift,
                                   self.outputs[0].zerop, self.outputs[0].qmin, self.outputs[0].qmax)

    self.outputs[0].betensor = output
    return self.outputs[0].betensor
