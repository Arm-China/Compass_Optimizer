# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.ops.select import *
import torch

register_optype('Where')


@op_register(OpType.Where)
def where_forward(self, *args):
    if len(self.inputs) > 1:
        return select_forward(self, *args)
    else:
        inp = self.inputs[0]
        out = self.outputs[0]
        indexes = torch.where(inp.betensor + inp.zerop)
        input_dim = inp.betensor.dim()
        valid_num = indexes[0].numel()
        total_num = inp.betensor.numel()
        invalid_num = total_num - valid_num
        y = torch.tensor([t.cpu().numpy() for t in indexes], device=inp.betensor.device).long()
        _, invalid_value = dtype2range(out.dtype)
        # arrange indexes like tf.where results
        # out.betensor = torch.nn.functional.pad(y.permute(1, 0), (0, 0, 0, inp.betensor.numel()-num), value=padding_value)
        first_invalid_num = min(1, invalid_num)
        invalid_tensor = torch.ones([first_invalid_num, input_dim], device=inp.betensor.device).long() * invalid_value

        other_invalid_num = max(0, invalid_num - 1)
        other_invalid_tensor = torch.zeros([other_invalid_num, input_dim], device=inp.betensor.device).long()

        out.betensor = torch.cat([y.permute(1, 0), invalid_tensor, other_invalid_tensor], dim=0)
        return out.betensor


@quant_register(OpType.Where)
def where_quantize(self, *args):
    if len(self.inputs) > 1:
        select_quantize(self, *args)
    else:
        inp = self.inputs[0]
        out = self.outputs[0]
        str_type = self.attrs['layer_top_type_original'][0]
        out.dtype = str2dtype(str_type)
        out.qbits = dtype2bits(out.dtype)
        out.scale = 1.0
        out.zerop = 0
        out.qinvariant = True
