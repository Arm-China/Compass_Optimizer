# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_ERROR

register_optype('BatchToSpaceND')


@op_register(OpType.BatchToSpaceND)
def batchtospaceNd(self, *args):
    block_size = self.params['block_size']
    crops = self.params['crops']

    inp = self.inputs[0].betensor

    if inp.ndim != 5:
        OPT_FATAL(f"{self}, currently input dim only supoort 5-dim, more dimensions will be supported in the future!")
    if self.inputs[0].ir_shape[0] != inp.shape[0]:
        OPT_ERROR(f"{self},batch size in calibratoin or metric dataset should be equal to batch size in IR")

    bs_z, bs_y, bs_x = block_size
    n, d, h, w, c = inp.shape

    y = inp.view(bs_z, bs_y, bs_x, n // (bs_z * bs_y * bs_x), d, h, w, c)  # ndhwc ->
    y = y.permute(3, 4, 0, 5, 1, 6, 2, 7).contiguous()  # n//(bs_z*bs_y*bs_x), d, bs_z, h, bs_y, w, bs_x, c
    y = y.view(n // (bs_z * bs_y * bs_x), d * bs_z, h * bs_y, w * bs_x, c)
    self.outputs[0].betensor = y[:, crops[0][0]: d * bs_z - crops[0][1], crops[1][0]: h * bs_y - crops[1][1],
                                 crops[2][0]: w * bs_x - crops[2][1], :]
    return self.outputs[0].betensor


@quant_register(OpType.BatchToSpaceND)
def batchtospaceNd_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.dtype = inp.dtype
    out.qinvariant = inp.qinvariant
