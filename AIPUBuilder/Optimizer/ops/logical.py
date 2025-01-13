# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *


@op_register(OpType.Logical)
def logical(self, *args):
    method_ = self.get_param("method").upper()
    supported_methods = ('NOT_EQUAL', 'EQUAL', 'GREATER', 'GREATER_EQUAL',
                         'LESS', 'LESS_EQUAL', 'AND', 'OR', 'NOT', 'XOR')
    if method_ not in supported_methods:
        OPT_FATAL('logical method "%s" is currently not supported' % (method_))
    inp = self.inputs[0].betensor
    out = self.outputs[0].betensor
    if method_ == 'NOT':
        inp = inp.type(torch.bool)
        out = torch.logical_not(inp)
    # method has 2 input
    else:
        inp1 = self.inputs[1]
        ref_val = inp1.betensor

        def set_can_detile(t):
            if t.pnode is None:
                can_detile = False
            else:
                pow_nods, count_root, count_constant = t.pnode.get_ancestors()
                can_detile = True if count_root > 0 and count_root == count_constant else False
            return can_detile
        inp0_can_detile = self.get_attrs('inp0_can_detile', optional=True, default_value=None)
        inp1_can_detile = self.get_attrs('inp1_can_detile', optional=True, default_value=None)
        if inp0_can_detile is None:
            inp0_can_detile = set_can_detile(self.inputs[0])
            self.attrs['inp0_can_detile'] = inp0_can_detile
        if inp1_can_detile is None:
            inp1_can_detile = set_can_detile(self.inputs[1])
            self.attrs['inp1_can_detile'] = inp1_can_detile
        if inp0_can_detile:
            inp = self.inputs[0].detile_betensor()
        if inp1_can_detile:
            ref_val = self.inputs[1].detile_betensor()

        if method_ in ('AND', 'OR', 'XOR'):
            inp = inp.type(torch.bool)
            ref_val = ref_val.type(torch.bool)
        elif self.quantized:
            scale0, scale1 = self.params["scale_value"]
            inp = (inp.int() + self.inputs[0].zerop) * scale0
            ref_val = (ref_val.int() + self.inputs[1].zerop) * scale1
        if method_ == 'NOT_EQUAL':
            out = ~torch.eq(inp, ref_val)
        if method_ == 'EQUAL':
            out = torch.eq(inp, ref_val)
        if method_ == 'GREATER':
            out = torch.gt(inp, ref_val)
        if method_ == 'GREATER_EQUAL':
            out = torch.greater_equal(inp, ref_val)
        if method_ == 'LESS':
            out = torch.less(inp, ref_val)
        if method_ == 'LESS_EQUAL':
            out = torch.less_equal(inp, ref_val)
        if method_ == 'AND':
            out = torch.logical_and(inp, ref_val)
        if method_ == 'OR':
            out = torch.logical_or(inp, ref_val)
        if method_ == 'XOR':
            out = torch.logical_xor(inp, ref_val)
    self.outputs[0].betensor = out.int()
    return self.outputs[0].betensor


@quant_register(OpType.Logical)
def logical_quantize(self, *args):
    if self.get_param("method").upper() not in ['AND', 'OR', 'NOT', 'XOR']:
        # two input may have different scale
        inp0 = self.inputs[0]
        inp1 = self.inputs[1]
        if inp0.qinvariant and inp1.qinvariant:
            scale0 = 1
            scale1 = 1
        else:
            if inp0.qinvariant != inp1.qinvariant:
                OPT_WARN('one input is quantize invariant and other one input is not, which may cause accuracy issue. layer_id=%s, %s' % (self.attrs['layer_id'], self.name),
                         workflow_name='quantize', op_name=str(self.type))
                if inp0.qinvariant and not inp1.qinvariant:
                    inp_scale_max = inp0.scale
                    inp_scale_min = inp1.scale
                else:  # inp1.qinvariant and not inp0.qinvariant:
                    inp_scale_max = inp1.scale
                    inp_scale_min = inp0.scale
            else:
                inp_scale_max = max(inp0.scale, inp1.scale)
                inp_scale_min = min(inp0.scale, inp1.scale)

            nfactor = min(32767, min((inp_scale_max / inp_scale_min)
                          * 8192, 32767) / (inp_scale_max / inp_scale_min))
            scale0 = (inp_scale_max / inp0.scale) * nfactor
            scale1 = (inp_scale_max / inp1.scale) * nfactor
        # if scale0 or scale1 is float32 1.0, int(scale0) is 0, which cause unexpected result
        if scale0 < 1:
            scale0 = 1
        if scale1 < 1:
            scale1 = 1
        self.params["scale_value"] = [int(scale0), int(scale1)]
        self.params["scale_type"] = [Dtype.UINT16]*2

    out = self.outputs[0]
    out.dtype = Dtype.INT8 if self.force_dtype_int else Dtype.UINT8
    out.scale = 1
    out.zerop = 0
    out.qbits = dtype2bits(out.dtype)
    out.qinvariant = True
    out.qmin, out.qmax = dtype2range(out.dtype)
