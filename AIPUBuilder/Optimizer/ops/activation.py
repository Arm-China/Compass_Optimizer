# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.ops.relu import *
from AIPUBuilder.Optimizer.ops.relu6 import *
from AIPUBuilder.Optimizer.ops.tanh import *
from AIPUBuilder.Optimizer.ops.sigmoid import *
from AIPUBuilder.Optimizer.ops.clip import *
from AIPUBuilder.Optimizer.ops.leakyrelu import *
from AIPUBuilder.Optimizer.ops.prelu import *
from AIPUBuilder.Optimizer.ops.elu import *
from AIPUBuilder.Optimizer.ops.selu import *
from AIPUBuilder.Optimizer.ops.crelu import *
from AIPUBuilder.Optimizer.ops.hardswish import *
from AIPUBuilder.Optimizer.ops.hardsigmoid import *
from AIPUBuilder.Optimizer.ops.softplus import *
from AIPUBuilder.Optimizer.ops.softsign import *
from AIPUBuilder.Optimizer.ops.silu import *
from AIPUBuilder.Optimizer.ops.mish import *
from AIPUBuilder.Optimizer.ops.celu import *
from AIPUBuilder.Optimizer.ops.thresholdrelu import *
from AIPUBuilder.Optimizer.ops.shrink import *
from AIPUBuilder.Optimizer.ops.gelu import *
from AIPUBuilder.Optimizer.ops.swish import *


def none_quantize(self, *args):
    pass


def none_activation(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    x = inp.betensor
    if self.quantized:
        do_shift = 0
        if "shift" in self.constants:
            do_shift = self.constants["shift"].betensor
        elif "shift_value" in self.params:
            do_shift = self.params["shift_value"]
        do_scale = 1
        if "scale" in self.constants:
            do_scale = self.constants["scale"].betensor
        elif "scale_value" in self.params:
            do_scale = self.params["scale_value"]
        x = linear_requantize(x + inp.zerop, do_scale, do_shift,
                              self.outputs[0].zerop, self.outputs[0].qmin, self.outputs[0].qmax)
    out.betensor = x
    return out.betensor


def unknown_activation(self, *args):
    '''this function mainly supports quant forward, which has activation method=unknown op.'''
    inp = self.inputs[0]
    out = self.outputs[0]
    if self.quantized:
        x = inp.betensor
        x = x - inp.qmin
        lut = self.constants["lut"].betensor
        x = torch.reshape(x, (-1,))
        y = lookup_lut_powerof2(x, lut, inp.qbits, False, dtype2bits(
            self.constants["lut"].dtype), is_signed(self.constants["lut"].dtype))
        out.betensor = torch.reshape(y, inp.betensor.shape)
    else:
        OPT_WARN(f"Activation op method=UNKNOWN donot support float forward, and the output.tensor will directly use input.tensor")
        out.betensor = inp.betensor
    return out.betensor


#                     lower_case_method_name        output_type_is_signed         forward_func     quantize_func
g_activation_method_supported = {'sigmoid':         (False,                       sigmoid,         sigmoid_quantize),
                                 'tanh':            (True,                        tanh,            tanh_quantize),
                                 'relu':            (False,                       relu,            relu_quantize),
                                 'relu6':           (False,                       relu6,           relu6_quantize),
                                 'crelu':           (False,                       crelu,           crelu_quantize),
                                 'clip':            (clip_out_signed,             clip,            clip_quantize),
                                 'leakyrelu':       (True,                        leakyrelu,       leakyrelu_quantize),
                                 'prelu':           (True,                        prelu,           prelu_quantize),
                                 'elu':             (True,                        elu,             elu_quantize),
                                 'selu':            (True,                        selu,            selu_quantize),
                                 'hardswish':       (True,                        hardswish,       hardswish_quantize),
                                 'hardsigmoid':     (hardsigmoid_out_signed,      hardsigmoid,     hardsigmoid_quantize),
                                 'softplus':        (False,                       softplus,        softplus_quantize),
                                 'softsign':        (True,                        softsign,        softsign_quantize),
                                 'silu':            (True,                        silu,            silu_quantize),
                                 'mish':            (True,                        mish,            mish_quantize),
                                 'celu':            (True,                        celu,            celu_quantize),
                                 'thresholdedrelu': (threshold_out_signed,        thresholdedrelu, thresholdedrelu_quantize),
                                 'shrink':          (True,                        shrink,          shrink_quantize),
                                 'gelu':            (True,                        gelu,            gelu_quantize),
                                 'swish':           (True,                        swish,           swish_quantize),
                                 'none':            (True,                        none_activation, none_quantize),
                                 'unknown':         (True,                        unknown_activation, none_quantize),
                                 }
with_activation_supported = ['none', 'clip', 'relu', 'relu6', 'leakyrelu', 'prelu']
with_activation_allow_merge_out_zerop_to_bias = ('none', 'clip', 'relu', 'relu6',)
assert {i for i in with_activation_supported}.issubset(g_activation_method_supported.keys()), OPT_FATAL(
    'OPT supported with_activation is not the subset of g_activation_method_supported')


@quant_register(OpType.Activation)
def activation_quantize(self, *args):
    method = self.get_param('method').lower()
    if method not in g_activation_method_supported:
        OPT_FATAL('activation method "%s" is currently not supported' % (method))
    func = g_activation_method_supported[method][2]
    func(self, *args)


@op_register(OpType.Activation)
def activation(self, *args):
    method = self.get_param('method').lower()
    if method not in g_activation_method_supported:
        OPT_FATAL('activation method "%s" is currently not supported' % (method))
    func = g_activation_method_supported[method][1]
    return func(self, *args)


def with_activation_out_is_signed(self):
    act_type = self.get_param('with_activation', optional=True, default_value='none').lower()
    if act_type not in with_activation_supported:
        (OPT_FATAL("id=%s, optype=%s, 'with_activation=%s', but with_activation only supported:%s." %
                   (self.attrs['layer_id'], str(self.type), act_type.upper(), str(with_activation_supported))))
    out_signed = g_activation_method_supported[act_type][0]
    if isinstance(out_signed, bool):
        return out_signed
    else:
        return out_signed(self)


def apply_with_activation_quantize(self, x_qinvariant, *args):

    _tensor_default_property = get_tensor_default_property()
    bak_inp_tensor_property = dict()
    for t in self.inputs:
        property = {}
        for p in _tensor_default_property:
            property.update({p: t.__getattribute__(p)})
        bak_inp_tensor_property.update({t.name: property})
    bak_out_tensor_property = dict()
    for t in self.outputs:
        property = {}
        for p in _tensor_default_property:
            property.update({p: t.__getattribute__(p)})
        bak_out_tensor_property.update({t.name: property})

    need_bak_params = {'shift_type': None,
                       'scale_type': None,
                       'scale_value': None,
                       'shift_value': None}
    for k, v in need_bak_params.items():
        if k in self.params:
            need_bak_params.update({k: self.params[k]})

    bak_shift_tensor = None
    if 'shift' in self.constants:
        bak_shift_tensor = self.constants["shift"].betensor
    bak_scale_tensor = None
    if 'scale' in self.constants:
        bak_scale_tensor = self.constants["scale"].betensor

    self.inputs[0].qinvariant = x_qinvariant

    activation = self.get_param('with_activation', optional=True, default_value='none').lower()
    if activation not in with_activation_supported:
        (OPT_FATAL("id=%s, optype=%s, 'with_activation=%s', but with_activation only supported:%s." %
                   (self.attrs['layer_id'], str(self.type), activation.upper(), str(with_activation_supported))))
    func = g_activation_method_supported[activation][2]
    func(self, *args)

    for k, v in need_bak_params.items():
        if k in self.params:
            if v is not None:
                self.params[k] = v
            else:
                self.params.pop(k)

    if 'shift' in self.constants:
        if bak_shift_tensor is not None:
            self.constants["shift"].betensor = bak_shift_tensor
        else:
            self.constants.pop('shift')
    if 'scale' in self.constants:
        if bak_scale_tensor is not None:
            self.constants["scale"].betensor = bak_scale_tensor
        else:
            self.constants.pop('scale')

    for t in self.inputs:
        for p in _tensor_default_property:
            t.__setattr__(p, bak_inp_tensor_property[t.name][p])
    for t in self.outputs:
        for p in _tensor_default_property:
            t.__setattr__(p, bak_out_tensor_property[t.name][p])


def apply_with_activation(self, x, x_scale, x_zerop, requant_scale, requant_shift, *args):
    bak_betensor = self.inputs[0].betensor
    bak_scale = self.inputs[0].scale
    bak_zerop = self.inputs[0].zerop
    self.inputs[0].betensor = x
    self.inputs[0].scale = x_scale
    self.inputs[0].zerop = x_zerop

    # this code is for inserting a placeholder tensor between convolution and with_activation op and for quantizing the with_activation op
    # if not self.quantized and self.get_param('with_activation', optional=True, default_value='none').lower() in ['clip']:
    #     if 'with_activation_placeholder_name' not in self.attrs:
    #         ph_len = len(self.placeholders)
    #         ph_name = self.name + '_placeholder_' + str(ph_len)
    #         self.attrs['with_activation_placeholder_name'] = ph_name
    #         stat_result = PyTensor(ph_name, x.cpu().numpy())
    #         self.placeholders.append(stat_result)
    #     else:
    #         placeholder_name = self.attrs['with_activation_placeholder_name']
    #         for p in self.placeholders:
    #             if p.name == placeholder_name:
    #                 p.betensor = x

    if self.quantized:
        bak_shift_value = 0
        if 'shift_value' in self.params:
            bak_shift_value = self.params['shift_value']
            self.params['shift_value'] = requant_shift
        bak_scale_value = 1
        if 'scale_value' in self.params:
            bak_scale_value = self.params['scale_value']
            self.params['scale_value'] = requant_scale
        bak_shift_tensor = 0
        if 'shift' in self.constants:
            bak_shift_tensor = self.constants["shift"].betensor
            self.constants["shift"].betensor = requant_shift
        bak_scale_tensor = 0
        if 'scale' in self.constants:
            bak_scale_tensor = self.constants["scale"].betensor
            self.constants["scale"].betensor = requant_scale

    act_type = self.get_param('with_activation', optional=True, default_value='none').lower()
    if act_type not in with_activation_supported:
        (OPT_FATAL("id=%s, optype=%s, 'with_activation=%s', but with_activation only supported:%s." %
                   (self.attrs['layer_id'], str(self.type), act_type.upper(), str(with_activation_supported))))
    bak_out_zerop = self.outputs[0].zerop
    # if act_type in with_activation_allow_merge_out_zerop_to_bias and 'biases' in self.constants:
    #     self.outputs[0].zerop = 0 # biases has absorbed out.zerop
    func = g_activation_method_supported[act_type][1]
    x = func(self, *args)
    self.outputs[0].zerop = bak_out_zerop
    if self.quantized:
        if 'shift_value' in self.params:
            self.params['shift_value'] = bak_shift_value
        if 'scale_value' in self.params:
            self.params['scale_value'] = bak_scale_value
        if 'shift' in self.constants:
            self.constants["shift"].betensor = bak_shift_tensor
        if 'scale' in self.constants:
            self.constants["scale"].betensor = bak_scale_tensor

    self.inputs[0].betensor = bak_betensor
    self.inputs[0].scale = bak_scale
    self.inputs[0].zerop = bak_zerop

    return x
