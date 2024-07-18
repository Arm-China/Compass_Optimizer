# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

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
        do_shift = self.get_ir_field(['shift', 'shift_value'], default_value=0)
        do_scale = self.get_ir_field(['scale', 'scale_value'], default_value=1)
        key_axis = out.key_axis
        x = linear_requantize(x + inp.broadcast_zerop, do_scale, do_shift, self.outputs[0].broadcast_zerop,
                              self.outputs[0].qmin, self.outputs[0].qmax, key_axis)
    out.betensor = x
    return out.betensor


def unknown_activation(self, *args):
    '''this function mainly supports quant forward, which has activation method=unknown op.'''
    inp = self.inputs[0]
    out = self.outputs[0]
    if self.quantized:
        x = inp.betensor.long()
        qmin = inp.qmin - (1 if (is_signed(inp.dtype) and abs(inp.qmin) == abs(inp.qmax)) else 0)
        x = x - qmin
        lut = self.constants["lut"].betensor
        x = torch.reshape(x, (-1,))
        y = lookup_lut_powerof2(x, lut, inp.qbits, False, dtype2bits(
            self.constants["lut"].dtype), is_signed(self.constants["lut"].dtype))
        out.betensor = torch.reshape(y, inp.betensor.shape)
    else:
        func = self.get_attrs('lambda_func', optional=True, default_value=None)
        if func is not None:
            out.betensor = func(inp.betensor)
        else:
            OPT_WARN(f"Activation op method=UNKNOWN does not support float forward, and the output.tensor will directly use input.tensor")
            out.betensor = inp.betensor
    return out.betensor


def unknown_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    out = self.outputs[0]
    out.qbits = q_bits_activation
    out_sign = self.get_attrs('out_signed', optional=True, default_value=True)
    dev = inp.betensor.device
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, out_sign)
    lsteps = 2 ** min(inp.qbits, int(self.get_attrs('lut_items_in_bits')))
    func = self.get_attrs('lambda_func', optional=True, default_value=None)
    mirror_offset = self.get_attrs('mirror_offset', optional=True, default_value=0.0)
    if func is not None:
        if is_signed(inp.dtype) and (-inp.qmin == inp.qmax):
            linspace = torch.linspace(inp.qmin, inp.qmax, steps=lsteps - 1, device=dev)
            linspace = torch.concat([torch.tensor([inp.qmin], device=dev), linspace])
        else:
            linspace = torch.linspace(inp.qmin, inp.qmax, steps=lsteps, device=dev)
        lut = linear_dequantize(linspace, inp.scale, inp.zerop)
        lut = func(lut) - linspace.new_tensor(mirror_offset)
        lut = linear_quantize_clip(lut, out.scale, out.zerop, out.qmin, out.qmax) + \
            torch.round(mirror_offset * out.scale)
        self.constants["lut"] = PyTensor(f"{self.name}/{func.__name__}_lut",
                                         lut.cpu().numpy().astype(dtype2nptype(out.dtype)))
        self.constants["lut"].dtype = out.dtype
    out.qinvariant = False


def unknown_approx(self, *args):
    inp = self.inputs[0]
    dev = inp.device
    approx_params = self.get_attrs('approx_params', optional=True, default_value=[0])
    method = int(approx_params[0] if len(approx_params) > 0 else 0)
    min_compatible_zhouyi_target = self.attrs["min_compatible_zhouyi_target"].upper()
    lut_items_in_bits = Target.aiff_lut_items_in_bits(min_compatible_zhouyi_target)
    if 1 == method and Target.optimized_target_level(min_compatible_zhouyi_target) >= 2:
        q_mode_activation = self.attrs["q_mode_activation"]
        use_dynamic_lut = len(approx_params) > 1 and approx_params[1] > 0
        bak_min = inp.min
        bak_max = inp.max
        set_min_max_func = self.get_attrs('set_min_max', optional=True, default_value=None)
        if set_min_max_func is not None:
            inp.min, inp.max = set_min_max_func(inp, use_dynamic_lut)
        out_sign = self.get_attrs('out_signed', optional=True, default_value=True)
        index_scale, index_offset, _, _, _ = get_linear_quant_params_from_tensor(
            inp, QuantMode.to_asymmetric(QuantMode.to_per_tensor(q_mode_activation)), lut_items_in_bits, out_sign)
        inp.min = bak_min
        inp.max = bak_max
        lut = linear_dequantize(torch.arange(0, 2**lut_items_in_bits, device=dev), index_scale, index_offset)
        value_offset = self.get_attrs('value_offset', optional=True, default_value=0)
        func = self.get_attrs('lambda_func', optional=True, default_value=None)
        if func is not None:
            lut = func(lut) + value_offset
        else:
            OPT_WARN(
                f"{self} Activation op method=UNKNOWN does not support float forward, and the output.tensor will directly use input.tensor")
        lut = to_fp24(lut)
        self.constants["lut"] = PyTensor(self.name + "/plh_lut", lut.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
        self.params['is_perf_mode'] = True
        self.params['lut_mode'] = 'NORMAL'
        self.params['index_scale_value'] = index_scale
        self.params['index_scale_type'] = Dtype.FP32
        self.params['index_offset_value'] = index_offset
        self.params['index_offset_type'] = Dtype.FP32
        self.params['value_offset_value'] = value_offset
        self.params['value_offset_type'] = Dtype.FP32
    else:
        # not suit for aiff, need use tpc to implement a high accuracy version
        self.params['is_perf_mode'] = False


#                     lower_case_method_name        output_type_is_signed         forward_func     quantize_func
g_activation_method_supported = {'sigmoid':         (False,                       sigmoid_forward, sigmoid_quantize),
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
                                 'unknown':         (True,                        unknown_activation, unknown_quantize),
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


def apply_with_activation(self, x, *args, **kargs):
    simulate_ahead_shift = False
    shift_bk = None
    for k, v in kargs.items():
        if k == 'aasrb':
            simulate_ahead_shift = True
            aasrb, bias = v
            do_shift = 0
            if "shift" in self.constants:
                do_shift = self.constants["shift"].betensor
                shift_bk = self.constants["shift"].clone()
            elif "shift_value" in self.params:
                do_shift = self.params["shift_value"]
            x, shift = aiff_ahead_shift_bias(x, do_shift, bias, int(aasrb))
            self.constants['shift'] = PyTensor('shift', torch.tensor(shift))
            break

    bak_betensor = self.inputs[0].betensor
    bk_inp_zp = self.inputs[0].zerop
    self.inputs[0].betensor = x
    self.inputs[0].zerop = args[0] if len(args) == 1 else 0

    act_type = self.get_param('with_activation', optional=True, default_value='none').lower()
    if act_type not in with_activation_supported:
        (OPT_FATAL("id=%s, optype=%s, 'with_activation=%s', but with_activation only supported:%s." %
                   (self.attrs['layer_id'], str(self.type), act_type.upper(), str(with_activation_supported))))
    bak_out_zp = self.outputs[0].zerop
    # if act_type in with_activation_allow_merge_out_zerop_to_bias and 'biases' in self.constants:
    #     self.outputs[0].zerop = 0 # biases has absorbed out.zerop
    func = g_activation_method_supported[act_type][1]
    x = func(self, *args)

    self.outputs[0].zerop = bak_out_zp
    self.inputs[0].zerop = bk_inp_zp
    self.inputs[0].betensor = bak_betensor

    if simulate_ahead_shift:
        if shift_bk is not None:
            self.constants['shift'] = shift_bk
        else:
            del self.constants['shift']
    return x
