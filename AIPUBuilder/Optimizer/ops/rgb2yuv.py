# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_WARN
import torch

register_optype('RgbToYuv')


@op_register(OpType.RgbToYuv)
def rgbToyuv(self, *args):
    inp = self.inputs[0].betensor  # [n,h,w,c]
    out = self.outputs[0]  # assume [n, h*w+h//2*w//2*2], TBD
    batch, height, width = inp.shape[:3]
    conversion = self.get_param('conversion', optional=False, default_value='BT709')
    format_ = self.get_param('format', optional=False, default_value='I420')
    # local_coe = {'BT709': [0, 0, 0, 0, 128, 128, 218, 732, 74, -118, -395, 512, 512, -465, -47],
    #              'BT601': [0, 0, 0, 0, 128, 128, 306, 601, 117, -173, -339, 512, 512, -465, -47],
    #              'SELF': self.get_param('coefficient')}
    local_coe = self.get_param('coefficient', optional=False)  # use param
    coes_shift = self.get_param('coefficient_shift', optional=False, default_value=10)

    # extract coes
    coes = torch.tensor(local_coe).to(inp.device).float()
    coe_in_offset = torch.reshape(coes[0:3], [3])  # constant, tile
    coe_out_offset = torch.reshape(coes[3:6], [3])  # constant, tile
    coe_calc = torch.reshape(coes[6:], [3, 3])

    img2 = nhwc2nchw(inp)  # nhwc->nchw

    # original impl:
    # yuvv = (torch.nn.functional.conv2d(img2+coe_in_offset, torch.tensor(coe_calc))+512) >>coes_shift + coe_out_offset

    # convert into conv style:
    bb2 = torch.matmul(-coe_in_offset, coe_calc) + coe_out_offset * 2 ** coes_shift + 512
    yuvv = torch.floor(torch.nn.functional.conv2d(img2, torch.reshape(
        coe_calc, [3, 3, 1, 1])) + bb2.reshape(3, 1, 1)) * 1 * 0.5 ** coes_shift

    # 444->420
    uuvv = torch.nn.functional.max_pool2d(yuvv[:, 1:, :, :], kernel_size=(1, 1), stride=(2, 2), padding=0,
                                          ceil_mode=False)

    # pack into i420
    yuvv_out = torch.cat([yuvv[:, 0, :, :].reshape(batch, -1), uuvv.reshape(batch, -1)],
                         dim=-1)  # shape=[h*w+h//2*w//2*2]

    if self.quantized:
        yuvv_out = torch.round(yuvv_out)
    yuvv_out = torch.clamp(yuvv_out, 0, 255)
    out.betensor = yuvv_out
    return out.betensor


@quant_register(OpType.RgbToYuv)
def rgbToyuv_quantize(self, *args):
    # param['bits'] =qbits
    qbits = 10  # self.get_param('bits', optional=False)
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
