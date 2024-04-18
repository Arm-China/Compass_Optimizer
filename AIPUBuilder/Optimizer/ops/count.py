# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from numpy.core.numeric import zeros_like
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *

register_optype('Count')


@op_register(OpType.Count)
# IR
# layer_type=Count
# layer_bottom=[mrcnn_detecion/mask_sorted_id_output]
# layer_bottom_shape=[[1,1000]]
# layer_bottom_type=[float32]
# layer_top=[mrcnn_detecion/histogram_output]
# layer_top_shape=[[1,81]]
# layer_top_type=[float32]
# range=[1,81]
# min=
# max=
# nbins=
# discrete=
# attention please, this maskrcnn histogram from up to down order
def countop(self, *args):
    # to match current maskrcnn float32 IR, and maskrcnn must be from big to small, so exchange min and max
    if 'nbins' not in self.params:
        min_value = torch.max(self.inputs[1].betensor)
        max_value = torch.min(self.inputs[1].betensor)
        bin = (min_value-max_value+1).int()
    else:
        bin = self.get_param('nbins')
        min_value = self.get_param('min')
        max_value = self.get_param('max')
    discrete = self.get_param('discrete', optional=True, default_value=True)
    if discrete == False:  # nbins means interval between min and max
        # this branch, pytorch api histc don't support
        # interval is [interval[i],interval[i+1]) format
        interval = torch.linspace(min_value, max_value, steps=bin+1)
        inp_betensors = self.inputs[0].betensor
        out = torch.zeros(self.outputs[0].ir_shape, device=self.outputs[0].betensor.device)
        for batch in range(int(self.inputs[0].shape[0])):
            for i in range(bin):
                ge = torch.ge(inp_betensors[batch].flatten(), interval[i])
                lt = torch.lt(inp_betensors[batch].flatten(), interval[i+1])
                out[batch, i] = (ge*lt).sum()
    else:

        # if self.inputs[0].shape[0] > 1:
        out = torch.zeros(self.outputs[0].ir_shape, device=self.outputs[0].betensor.device)
        inp_betensors = self.inputs[0].betensor
        for i in range(int(self.inputs[0].shape[0])):
            if max_value > min_value:  # from small to big
                out[i] = torch.histc(inp_betensors[i].float(), bin, min_value, max_value)
            else:
                # from big to small
                out[i] = torch.flip(torch.histc(inp_betensors[i], bin, max_value, min_value), [0])
        self.outputs[0].betensor = out
        # else:
        #     inp_betensors = self.inputs[0].betensor
        #     if max > min:  #from small to big
        #         out = torch.histc(inp_betensors.float(),bin,min,max)
        #     else:
        #         #from big to small
        #         out = torch.flip(torch.histc(inp_betensors,bin,max,min),[0])
    self.outputs[0].betensor = out.reshape(self.outputs[0].ir_shape)
    return self.outputs[0].betensor


@quant_register(OpType.Count)
def countop_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    min_value = self.get_param('min')
    max_value = self.get_param('max')
    self.params['min'] = (min_value * inp.scale).round().tolist()[0]
    self.params['max'] = (max_value * inp.scale).round().tolist()[0]
    out.qbits = max(16, inp.qbits)
    out.dtype = bits2dtype(out.qbits, False or self.force_dtype_int)
    out.scale = torch.ones_like(inp.scale)
    out.zerop = inp.zerop
    out.qmin, out.qmax = dtype2range(out.dtype)
    out.qinvariant = True
