# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_ERROR


# @op_register(OpType.Pad)
def pad(self, *args):
    out = self.outputs[0]
    pad_mode = self.get_param('mode').lower()
    pad_value = 0 if pad_mode != 'constant' else self.get_param('constant_value')

    pads = torch.tensor(self.get_param('pads'))

    inp = self.inputs[0].betensor
    # normalize to 4 dims
    ndim = inp.ndim
    for i in range(ndim, 4, 1):
        inp = inp.unsqueeze(0)
    inp = nhwc2nchw(inp)
    paddings = list(torch.zeros(inp.ndim*2).numpy())
    # from IR's padding layout is [N,H,W,C],so need convert to pytorch padding layout format [W,H,C,N]
    dim_ind = (4-pads.shape[0]+1)//2
    paddings[0:2] = pads[2-dim_ind, :]
    paddings[2:4] = pads[1-dim_ind, :]
    # dims is 3 or 4
    if pads.shape[0] > 2:
        paddings[4:6] = pads[3-dim_ind, :]
    # dims is 4
    if pads.shape[0] > 3:
        paddings[6:8] = pads[0, :]
    # for i in range(0, len(paddings), 2):
    #     tmp = paddings[i]
    #     paddings[i] = paddings[i + 1]
    #     paddings[i + 1] = tmp
    if pad_mode == 'constant':
        # padding dims layout format is[W,H,C,N] corresponding to data's layout[N,C,H,W]
        out.betensor = nchw2nhwc(torch.nn.functional.pad(inp, paddings, mode=pad_mode, value=pad_value))
    else:
        # Reflect padding is only implemented for padding the last 2 dimensions of 4D input tensor,
        # or the last dimension of 3D input tensor
        if len(paddings) != 2*inp.ndim:
            OPT_FATAL("each dim need have a padding,padding format is such as [[0, 0], [1, 1], [1, 1], [0, 0]]")
        # paddings layout w,h,c,n, data layout is [n,c,h,w]
        padded_hw = torch.nn.functional.pad(inp, paddings[0:4], mode=pad_mode, value=pad_value)
        # to pad data's [n,c] dim, data layout is converted from [n,c,h,w] to [h,w,n,c]
        shift_nc_to_hw = padded_hw.permute(2, 3, 0, 1)
        shift_nc_to_hw = torch.nn.functional.pad(shift_nc_to_hw, paddings[4:8], mode=pad_mode, value=pad_value)
        # now shift_nc_to_hw data layout is [h,w,n,c], so neet convert to n,h,w,c
        out.betensor = shift_nc_to_hw.permute(2, 0, 1, 3).reshape(out.betensor.shape)
    return out.betensor


@op_register(OpType.Pad)
def pad_forward(self, *args):
    inp = self.inputs[0].betensor
    outt = self.outputs[0]
    pad_mode = self.get_param('mode').lower()
    pad_value = 0 if pad_mode != 'constant' else self.get_param('constant_value')
    paddings = self.get_param('pads')

    def list_unpack(l):
        import functools
        return functools.reduce(lambda x, y: x + y, l)

    if pad_mode == 'constant':
        torch_paddings = paddings[::-1]  # compass IR:nhwc, torch: cwhn when input.ndims == 4
        torch_paddings = list_unpack(torch_paddings)
        out = torch.nn.functional.pad(inp, torch_paddings, mode=pad_mode, value=pad_value)
    elif pad_mode in ['symmetric', 'reflect']:
        import numpy as np
        inp_np = inp.cpu().numpy()
        out = np.pad(inp_np, paddings, mode=pad_mode)
        out = torch.tensor(out, device=inp.device)
    else:
        out = torch.zeros(outt.betensor.shape)
        OPT_ERROR(f"mode of Pad op now only support ['CONSTANT', 'REFLECT', 'SYMMETRIC'], but now is {pad_mode}")

    outt.betensor = out
    return outt.betensor


@quant_register(OpType.Pad)
def pad_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qmin = inp.qmin
    out.qmax = inp.qmax
    out.qinvariant = inp.qinvariant
    if 'constant_value' in self.params:
        pvalue = self.params['constant_value']
        qvalue = linear_quantize_clip(pvalue, out.scale, out.zerop, out.qmin, out.qmax)
        self.params['constant_value'] = qvalue.item()
