# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_ERROR, OPT_FATAL

register_optype('SlotUpdate')


@op_register(OpType.SlotUpdate)
def slotupdate_forward(self, *args):
    inputs = self.inputs
    if len(inputs) - 1 != self.inputs[0].ir_shape[1]:
        OPT_ERROR(
            f"{self}:please check the slotupdate IR, the len(inputs) - 1(={len(inputs) - 1}) should be equal to inputs[0].shape[1]={self.inputs[0].ir_shape[1]}")

    self.outputs[0].betensor = torch.zeros(self.outputs[0].ir_shape).to(self.inputs[0].device).float()
    bs = self.inputs[0].betensor.shape[0]
    inp0_betensor = self.inputs[0].betensor.float() + self.inputs[0].zerop
    for b in range(bs):
        for i, inp_t in enumerate(self.inputs[1:]):
            idx = inp_t.betensor.long().cpu().numpy().tolist()
            for k, id in enumerate(idx):
                self.outputs[0].betensor[b, id, :] += inp0_betensor[b, i, :inp_t.betensor.shape[0]][k]

    if self.quantized:
        shift = self.params["shift_value"]
        scale = self.params["scale_value"]
        self.outputs[0].betensor = linear_requantize(
            self.outputs[0].betensor, scale, shift, self.outputs[0].zerop, self.outputs[0].qmin, self.outputs[0].qmax)

    return self.outputs[0].betensor


@quant_register(OpType.SlotUpdate)
def slotupdate_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    shift_type = SHIFT_DTYPE

    q_bits_activation = self.attrs["q_bits_activation"]
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation):
        OPT_FATAL("Currently not support per-channel quantization of activations")
    out_signed = is_signed(inp.dtype) or self.force_dtype_int
    if inp.qinvariant:
        out.scale = 1
        out.zerop = 0
        out.qbits, out.dtype = range2dtype(
            out.extrema_min, out.extrema_max, force_int=out_signed)
        out.qinvariant = True
        out.qmin, out.qmax = dtype2range(out.dtype)
        do_scale, do_shift = 1, 0
        do_scale_type = bits2dtype(out.qbits, False)
    else:
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, q_bits_activation, is_signed=out_signed)
        out.qbits = q_bits_activation
        out.qinvariant = inp.qinvariant
        local_scale = out.scale / inp.scale
        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(local_scale,
                                           mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)

    self.params["shift_value"] = int(do_shift)
    self.params["shift_type"] = shift_type
    self.params["scale_value"] = int(do_scale)
    self.params["scale_type"] = do_scale_type
