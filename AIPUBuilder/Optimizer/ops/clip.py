# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import OPT_ERROR
from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.utils.quant_tool_utils import *


@op_register(OpType.Clip)
def clip(self, *args):
    inp_t = self.inputs[0].betensor
    out_max, out_min = min(self.get_param('clip_max'), OPT_INT_MAX), max(float(self.get_param('clip_min')), OPT_INT_MIN)
    if not self.quantized:
        out_t = torch.clamp(inp_t, out_min, out_max)
    else:
        if "shift" not in self.constants:
            doshift = self.params["shift_value"]
            doscale = self.params["scale_value"]
        else:
            doshift = self.constants["shift"].betensor
            doscale = self.constants["scale"].betensor
        i_zp, o_zp = self.inputs[0].zerop, self.outputs[0].zerop
        out = inp_t + i_zp

        # rescale to 16bits
        # then do the clamp
        relay_bits = 8 if self.outputs[0].qbits <= 8 else 0
        out_t = linear_requantize(
            out, doscale, doshift - relay_bits, 0, out_min, out_max)
        # finally scale to the out.scale
        out_t = linear_requantize(
            out_t, 1, relay_bits, self.outputs[0].zerop, self.outputs[0].qmin, self.outputs[0].qmax)

    self.outputs[0].betensor = out_t
    return out_t


@quant_register(OpType.Clip)
def clip_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    q_mode_activation = self.attrs['q_mode_activation']
    q_bits_activation = self.attrs['q_bits_activation']
    clip_min = self.get_param('clip_min')
    clip_max = self.get_param('clip_max')
    if clip_min > clip_max:
        OPT_ERROR('The "clip_min" must not greater than "clip_max" in layer_type=%s, layer_id=%s, layer_name=%s' % (
            str(self.type), str(self.attrs['layer_id']), str(self.name)))
    dev = inp.device
    out.min = min(max(out.min, torch_tensor(clip_min, device=dev)), torch_tensor(clip_max, device=dev))
    out.max = max(min(out.max, torch_tensor(clip_max, device=dev)), torch_tensor(clip_min, device=dev))
    if None != out.min_key_axis:
        out.min_key_axis = torch.min(torch.max(out.min_key_axis, torch.ones_like(
            out.min_key_axis)*clip_min), torch.ones_like(out.min_key_axis)*clip_max)
        out.max_key_axis = torch.max(torch.min(out.max_key_axis, torch.ones_like(
            out.max_key_axis)*clip_max), torch.ones_like(out.min_key_axis)*clip_min)

    out_signed = False if clip_min >= 0 else True
    out_signed = out_signed or self.force_dtype_int
    out.qinvariant = inp.qinvariant
    if inp.qinvariant:
        out.scale = 1.
        out.zerop = 0
        out.qbits = inp.qbits
        out.dtype = inp.dtype
    else:
        out.qbits = q_bits_activation
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, q_bits_activation, is_signed=out_signed)
    local_rescale = out.scale / inp.scale
    do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(
        local_rescale, mult_bits=out.qbits, force_shift_positive=self.force_shift_positive)
    self.params['scale_value'] = int(do_scale)
    self.params['shift_value'] = int(do_shift)
    self.params['scale_type'] = do_scale_type
    self.params['shift_type'] = do_shift_type

    cscale = out.scale
    czerop = out.zerop
    if out.qbits <= 8:
        cscale = out.scale * (2**8)
        czerop = 0
    t_qmin, t_qmax = bits2range(32, is_signed=True)
    cmax = linear_quantize_clip(
        clip_max, cscale, czerop, t_qmin, t_qmax).item()
    cmin = linear_quantize_clip(
        clip_min, cscale, czerop, t_qmin, t_qmax).item()
    self.params['clip_max'] = int(max(t_qmin, min(t_qmax, cmax)))
    self.params['clip_min'] = int(max(t_qmin, min(t_qmax, cmin)))


def clip_out_signed(self):
    clip_min = self.get_param('clip_min')
    return False if clip_min >= 0 else True
