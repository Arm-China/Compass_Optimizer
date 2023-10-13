# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_WARN
import torch

register_optype('YuvToRgb')


@op_register(OpType.YuvToRgb)
def yuvTorgb(self, *args):
    inp = self.inputs[0].betensor  # assume [n, h*w+h//2*w//2*2], TBD
    out = self.outputs[0]  # [n,h,w,c]
    batch = self.current_batch_size if self.current_batch_size != 0 else inp.shape[0]
    height = self.outputs[0].ir_shape[1]
    width = self.outputs[0].ir_shape[2]
    conversion = self.get_param('conversion', optional=True, default_value='BT709')
    format_ = self.get_param('format', optional=True, default_value='I420')
    # local_coe = {'BT709': [0, 128, 128, 0, 0, 0, 256, 0, 403, 256, -48, -120, 256, 475, 0],
    #              'BT601': [0, 128, 128, 0, 0, 0, 256, 0, 359, 356, -88, -183, 256, 454, 0],
    #              'SELF': self.get_param('coefficient')}
    local_coe = self.get_param('coefficient', optional=False)  # use param
    coes_shift = self.get_param('coefficient_shift', optional=False)

    # coefficients
    coes = torch.tensor(local_coe).to(inp.device).float()
    coe_in_offset = torch.reshape(coes[0:3], [3])
    coe_out_offset = torch.reshape(coes[3:6], [3])
    coe_calc = torch.reshape(coes[6:], [3, 3])

    # Extract yuv data from inp
    yy = inp[:, :height * width].reshape([batch, 1, height, width])
    uu = inp[:, height * width: height * width + height // 2 * width // 2].reshape([batch, 1, height // 2, width // 2])
    vv = inp[:, height * width + height // 2 * width // 2:].reshape([batch, 1, height // 2, width // 2])
    uv = torch.cat([uu, vv], 1)

    uv444 = torch.nn.functional.interpolate(torch.tensor(uv), size=[height, width], mode='nearest')
    # 420->422
    for h in range(1, height - 1):
        if h % 2 == 1:
            uv444[:, :, h, :] = torch.round(uv444[:, :, h - 1, :] / 2 + uv444[:, :, h + 1, :] / 2)
    # 422->444
    uv444_2 = torch.zeros_like(uv444)
    for w in range(1, width - 1):
        if w % 2 == 1:
            uv444_2[:, :, :, w] = torch.round((uv444[:, :, :, w - 1] * 3 + uv444[:, :, :, w + 1] * 1) * 0.5**2)
        else:
            uv444_2[:, :, :, w] = torch.round((uv444[:, :, :, w - 1] * 1 + uv444[:, :, :, w + 1] * 3) * 0.5**2)
    uv444[:, :, :, 1:-1] = uv444_2[:, :, :, 1:-1]
    yuv = torch.cat([yy, uv444], 1)  # nchw

    coe_calc = coe_calc.permute(1, 0)
    rgb = (torch.matmul(nchw2nhwc(yuv)-coe_in_offset, coe_calc) + 128)*0.5**coes_shift + coe_out_offset

    rgb = torch.clamp(torch.round(rgb), 0, 255)
    out.betensor = rgb
    return out.betensor


@quant_register(OpType.YuvToRgb)
def yuvTorgb_quantize(self, *args):
    # param['bits'] =qbits
    qbits = 8  # self.get_param('bits', optional=False)
    local_coe = self.get_param('coefficient', optional=False)
    q_coe = [round(c*2**qbits) for c in local_coe[6:]]
    local_coe[6:] = q_coe

    # generate params: format/coe_values
    conversion = self.get_param('conversion', optional=True, default_value='BT709')
    format_ = self.get_param('format', optional=True, default_value='I420')
    self.params['coefficient'] = local_coe
    self.params['coefficient_shift'] = qbits
    self.params['coefficient_dtype'] = 'int16'
    self.params['conversion'] = conversion
    self.params['format'] = format_

    inp = self.inputs[0]
    out = self.outputs[0]
    out.qinvariant = inp.qinvariant
    out.scale = 1.
    out.qbits = self.get_param('bits', optional=False)
    if self.force_dtype_int:
        out.dtype = Dtype.INT8
        out.zerop = 128
    else:
        out.dtype = Dtype.UINT8
        out.zerop = 0
