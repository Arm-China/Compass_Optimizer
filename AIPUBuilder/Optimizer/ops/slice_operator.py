# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.stridedslice import *


@op_register(OpType.Slice)
def slice_forward(self, *args):
    dynamic_slice = len(self.inputs) > 1
    if dynamic_slice:
        steps = []
        begin = []
        end = []
        dims = len(self.inputs[0].shape)
        axes = self.inputs[3].betensor
        axes = torch.where(axes < torch.zeros_like(axes), axes+dims, axes).clamp(0, dims-1).tolist()
        for d in range(dims):
            predefined = False
            for ad in range(len(axes)):
                if d == axes[ad]:
                    steps.append(self.inputs[4].betensor[ad])
                    begin.append(self.inputs[1].betensor[ad])
                    end.append(self.inputs[2].betensor[ad])
                    predefined = True
                    break
            if not predefined:
                steps.append(1)
                begin.append(0)
                end.append(self.inputs[0].shape[d])
        self.params['strides'] = steps
        self.params['begin'] = begin
        self.params['end'] = end
    stridedslice(self, *args)
    if dynamic_slice:
        self.params.pop('strides')
        self.params.pop('begin')
        self.params.pop('end')
    if len(self.outputs) > 1:
        self.outputs[1].betensor = torch.tensor(self.outputs[0].betensor.shape, device=self.outputs[0].betensor.device)
        return self.outputs[0].betensor, self.outputs[1].betensor
    else:
        return self.outputs[0].betensor


@quant_register(OpType.Slice)
def slice_quantize(self, *args):
    if len(self.outputs) > 1:
        # record the real shape of self.outputs[0]
        out = self.outputs[1]
        out.scale = 1.
        out.zerop = 0
        out.qbits = dtype2bits(out.ir_dtype)
        out.dtype = out.ir_dtype
        out.qinvariant = True
        self.attrs["q_mode_activation"] = QuantMode.to_per_tensor(self.attrs["q_mode_activation"])
    stridedslice_quantize(self, *args)
