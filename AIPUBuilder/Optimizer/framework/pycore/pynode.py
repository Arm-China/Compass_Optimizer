# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3

__all__ = [
    "PyNode",
    "get_op_dtype_spec",
]


class FeldsDict(dict):
    def clone(self):
        import copy
        d = self.__class__()
        for k, v in self.items():
            d[k] = copy.deepcopy(v)
        return d


class PyNode:
    # __slots__ = ('name', 'type', 'params', 'attrs', 'constants', 'inputs',
    #              'outputs', 'parents', 'children', 'placeholders', 'graph')

    def __init__(self, name, type) -> None:
        from AIPUBuilder.Optimizer.framework.pycore.pytype import OpTypeValue
        self.name = str(name)
        self.type = OpTypeValue(str(type))
        self.params = FeldsDict()  # IR fields
        self.attrs = FeldsDict()  # config for dev usage
        self.constants = {}  # weights, biases
        self.inputs = ()
        self.outputs = ()
        self.parents = ()
        self.children = ()
        self.placeholders = []  # internal Tensors
        self.graph = None  # the belonged graph

    def clone(self, name=None):
        import copy
        if name is None:
            name = self.name + '_clone'
        n = self.__class__(name, self.type)
        for k, v in self.params.items():
            n.params[k] = copy.deepcopy(v)
        for k, v in self.attrs.items():
            n.attrs[k] = copy.deepcopy(v)
        for k, v in self.constants.items():
            n.constants[k] = v.clone(v.name)
        for t in self.inputs:
            n.add_input(t.clone(t.name))
        for t in self.outputs:
            n.add_output(t.clone(t.name))
        for t in self.placeholders:
            n.placeholders.append(t.clone(t.name))
        n.graph = None
        return n

    def add_input(self, t, idx=-1):
        k = idx if idx >= 0 else len(self.inputs) + 1 + idx
        k = max(0, min(k, len(self.inputs)))
        self.inputs = self.inputs[:k] + (t, ) + self.inputs[k:]
        if self.graph:
            # update edges' relationship in corresponding graph
            self.graph.init_networkx()

    def add_output(self, t, idx=-1):
        k = idx if idx >= 0 else len(self.outputs) + 1 + idx
        k = max(0, min(k, len(self.outputs)))
        self.outputs = self.outputs[:k] + (t, ) + self.outputs[k:]
        if self.graph:
            # update edges' relationship in corresponding graph
            self.graph.init_networkx()

    def remove_input(self, t):
        flag = False
        idx = None
        for i, inp in enumerate(self.inputs):
            if inp.name == t.name:
                flag = True
                idx = i
        if flag:
            self.inputs = self.inputs[: idx] + self.inputs[idx+1:]
            if self.graph:
                # update edges' relationship in corresponding graph
                self.graph.init_networkx()
        return idx

    def remove_output(self, t):
        flag = False
        idx = None
        for i, out in enumerate(self.outputs):
            if out.name == t.name:
                flag = True
                idx = i
        if flag:
            self.outputs = self.outputs[: idx] + self.outputs[idx+1:]
            if self.graph:
                # update edges' relationship in corresponding graph
                self.graph.init_networkx()
        return idx

    def replace_input_temporarily(self, idx, t):
        '''
        replace one of inputs tensors temporarily for customized usage,
        remember to replace back, because this option just replace corresponding tensor
        without change correspoding graph
        '''
        k = idx if idx >= 0 else len(self.inputs) + idx
        k = max(0, min(k, len(self.inputs) - 1))
        old = self.inputs[k]
        self.inputs = self.inputs[: k] + (t,) + self.inputs[k+1:]
        return old

    def replace_output_temporarily(self, idx, t):
        '''
        replace one of outputs tensors temporarily for customized usage,
        remember to replace back, because this option just replace corresponding tensor
        without change correspoding graph
        '''
        k = idx if idx >= 0 else len(self.outputs) + idx
        k = max(0, min(k, len(self.outputs) - 1))
        old = self.outputs[k]
        self.outputs = self.outputs[: k] + (t,) + self.outputs[k+1:]
        return old

    def get_param(self, key, *, optional=False, default_value=None):
        from AIPUBuilder.Optimizer.logger import OPT_DEBUG, OPT_WARN, OPT_FATAL
        if key in self.params:
            return self.params[key]
        elif optional:
            OPT_DEBUG('Optional param "%s" not exist in layer_id=%s, %s, %s, use default value "%s".' % (
                key, self.attrs.get('layer_id', "-1"), str(self.type), self.name, str(default_value)), log_once=True)
            return default_value
        else:
            OPT_FATAL('Required param "%s" not exist in layer_id=%s, %s, %s, please check IR.' % (
                key, self.attrs.get('layer_id', "-1"), str(self.type), self.name))
            return None

    def get_attrs(self, key, *, optional=False, default_value=None):
        from AIPUBuilder.Optimizer.logger import OPT_DEBUG, OPT_WARN, OPT_FATAL
        if key in self.attrs:
            return self.attrs[key]
        elif optional:
            OPT_DEBUG('Optional attrs "%s" not exist in layer_id=%s, %s, %s, use default value "%s".' % (
                key, self.attrs.get('layer_id', "-1"), str(self.type), self.name, str(default_value)), log_once=True)
            return default_value
        else:
            OPT_FATAL('Required attrs "%s" not exist in layer_id=%s, %s, %s, please check IR.' % (
                key, self.attrs.get('layer_id', "-1"), str(self.type), self.name))
            return None

    def get_constant(self, key):
        from AIPUBuilder.Optimizer.logger import OPT_FATAL
        if key in self.constants:
            return self.constants[key]
        else:
            (OPT_FATAL('Required constant data "%s" not exist in layer_id=%s, %s, %s, please check IR.' %
                       (key, self.attrs.get('layer_id', "-1"), str(self.type), self.name)))
            return None

    def forward(self, *args):
        from AIPUBuilder.Optimizer.framework import OP_DICT
        from AIPUBuilder.Optimizer.utils.quant_tool_utils import linear_dequantize, linear_quantize_clip
        from AIPUBuilder.Optimizer.logger import OPT_WARN, OPT_ERROR, OPT_FATAL, OPT_DEBUG
        ret = None

        def _node_params_replace(dst_dict, src_dict, replace_keys=None):
            if replace_keys != None:
                for r_key in replace_keys:
                    if r_key in src_dict.keys():
                        dst_dict[r_key] = src_dict[r_key]
                    elif r_key in dst_dict.keys():
                        dst_dict.pop(r_key)
                    else:
                        OPT_WARN('the replace_keys:%s is not in src_dict:%s and node.params.' % (
                            r_key, src_dict.__name__))
                return None
            else:
                replace_keys = []
                for k, v in src_dict.items():
                    dst_dict[k] = src_dict[k]
                    replace_keys.append(k)
                return replace_keys

        if self.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        # decide wether to change quantized flag for fake quantization debug usage
        quant_state = self.quantized
        constants_betensor_backup = None
        params_backup = None
        replace_keys = None
        # copy_constants_to(self.constants, constants_backup)
        if True == self.quantized:
            if 'debug_fake_quantize' in self.attrs and self.attrs['debug_fake_quantize']:
                for inp in self.inputs:
                    if 0 == inp.debug_flag:
                        inp.betensor = linear_dequantize(
                            inp.betensor, inp.scale, inp.zerop)
                        inp.debug_flag = 1
                self.quantized = False
                constants_betensor_backup = {}
                params_backup = self.params.clone()
                replace_keys = _node_params_replace(
                    self.params, self.attrs['params_original'])
                for k, t in self.attrs['constants_betensor_original'].items():
                    constants_betensor_backup[k] = self.constants[k].betensor.clone(
                    )
                    self.constants[k].betensor = t.clone()
            else:
                for inp in self.inputs:
                    if 0 != inp.debug_flag:
                        inp.betensor = linear_quantize_clip(
                            inp.betensor, inp.scale, inp.zerop, inp.qmin, inp.qmax)
                        inp.debug_flag = 0
        # call OP's forward(), backup its inputs first in case that inp.betensor be modified in forward function
        maintained_inp_betensors = []
        for ii, iinp in enumerate(self.inputs):
            maintained_inp_betensors.append(iinp.betensor.clone())
        ret = OP_DICT[self.type](self, *args)
        for ii, iinp in enumerate(self.inputs):
            iinp.betensor = maintained_inp_betensors[ii]
        # restore quantized flag and constants
        self.quantized = quant_state
        if None != constants_betensor_backup:
            for k, t in constants_betensor_backup.items():
                self.constants[k].betensor = t.clone()
        if None != params_backup:
            _ = _node_params_replace(self.params, params_backup, replace_keys)
        if True == self.quantized:
            if 'debug_fake_quantize' in self.attrs and self.attrs['debug_fake_quantize']:
                for out in self.outputs:
                    out.debug_flag = 1
            else:
                for out in self.outputs:
                    out.debug_flag = 0

        for t in self.outputs:
            dshape = t.betensor.shape
            sshape = dshape
            if t.ir_shape:
                sshape = t.ir_shape
            else:
                OPT_WARN('IR Shape Info of Tensor: %s in layer_id=%s %s is lost' % (
                    t.name, str(self.attrs.get('layer_id', -1)), str(self.type)))
            flag = False
            if len(list(sshape)) != len(list(dshape)):
                flag = True
            else:
                batch_size_in_IR = self.get_attrs("batch_size_in_IR", optional=True, default_value=1)
                st = 1 if batch_size_in_IR > 0 else 0
                for i in range(st, len(list(sshape))):
                    if sshape[i] != dshape[i]:
                        flag = True
                        break
            if flag:
                OPT_WARN('Get inconformity tensor shape %s with the original IR %s, layer_id=%s, tensor_name=%s' % (
                    str(dshape), str(sshape), self.attrs.get('layer_id', "-1"), self.name), op_name=str(self.type))

        return ret

    def quantize(self, *args, **kwargs):
        ret = None
        from AIPUBuilder.Optimizer.framework import QUANT_OP_DICT, OpType
        from AIPUBuilder.Optimizer.utils.dtype_utils import is_signed, dtype2str
        from AIPUBuilder.Optimizer.logger import OPT_FATAL, OPT_DEBUG, OPT_ERROR, OPT_WARN

        if self.type not in QUANT_OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))
        unquantifiable = self.get_param(
            'unquantifiable', optional=True, default_value=False)
        # first set properties that must be decided during quantization
        # then backup the constants' betensor
        key_tensors = []
        for t in self.outputs:
            key_tensors.append(t)
        self.attrs['constants_betensor_original'] = {}
        # self.attrs['params_original'] = self.params_clone()
        self.attrs['params_original'] = self.params.clone()

        for k, t in self.constants.items():
            key_tensors.append(t)
            self.attrs['constants_betensor_original'][k] = t.betensor.clone()
        for t in self.placeholders:
            key_tensors.append(t)
        if not unquantifiable:
            for t in key_tensors:
                t.dtype = None
                t.scale = None
                t.zerop = None
                t.qbits = None
                t.qmin = None
                t.qmax = None
                t.qinvariant = None
        ####################
        # replace quantized source weights if adaround optimization was applied
        if 'weights' in self.constants and 'adaround_weights' in self.attrs:
            bkey = self.attrs['q_bits_weight']
            qnw = self.constants['weights']
            if bkey in self.attrs['adaround_weights'].keys():
                OPT_DEBUG('Adaround optimization was applied on layer_id=%s, %s, %s' %
                          (str(self.attrs['layer_id']), str(self.type), self.name))
                qnw.betensor = self.attrs['adaround_weights'][bkey]
        ####################
        # then do the quantization
        ret = QUANT_OP_DICT[self.type](self, *args, **kwargs)

        # finnaly check properties that must be decided during quantization
        self.attrs['quantization_info'] = {}
        t_q_dict = {}
        for t in (list(self.constants.values()) + list(self.outputs)):
            if t not in key_tensors:
                continue
            if not unquantifiable:
                if t.dtype == None:

                    OPT_FATAL('during quantization "%s" must decide "dtype" of "%s" tensor' % (
                        str(self.type), t.name))
                if t.scale == None:
                    OPT_FATAL('during quantization "%s" must decide "scale" of "%s" tensor' % (
                        str(self.type), t.name))
                if t.zerop == None:
                    OPT_FATAL('during quantization "%s" must decide "zerop" of "%s" tensor' % (
                        str(self.type), t.name))
                if t.qbits == None:
                    OPT_FATAL('during quantization "%s" must decide "qbits" of "%s" tensor' % (
                        str(self.type), t.name))
                if t.qmin == None:
                    t.qmin = -2 ** (t.qbits - 1) if is_signed(t.dtype) else 0
                    OPT_DEBUG('tensor "%s" using default qmin, cause not explicitly set in quantize func' %
                              t.name, workflow_name='quantize', op_name=str(self.type), log_once=True)
                if t.qmax == None:
                    t.qmax = 2 ** (t.qbits - 1) - \
                        1 if is_signed(t.dtype) else 2 ** t.qbits - 1
                    OPT_DEBUG('tensor "%s" using default qmax, cause not explicitly set in quantize func' %
                              t.name, workflow_name='quantize', op_name=str(self.type), log_once=True)
                if t.qinvariant == None:
                    OPT_FATAL('during quantization "%s" must decide "qinvariant" of "%s" tensor' % (
                        str(self.type), t.name))
            t_q_dict.update({
                t.name: {
                    'scale': str(t.scale),
                    'zerop': str(t.zerop),
                    'qbits': str(t.qbits),
                    'dtype': str(t.dtype),
                    'qmin': str(t.qmin),
                    'qmax': str(t.qmax),
                    'qinvariant': str(t.qinvariant),
                }
            })
        self.attrs.update({'quantization_info': t_q_dict})

        # check if key tensors were quantized with q_bits_xx, else log info
        def check_qbits_log(tname, requre_bits, actual_bits):
            if actual_bits < requre_bits:
                OPT_WARN('layer_id=%s, layer_type=%s : requred bits for tensor "%s" is %d,'
                         ' but actually got %d, which may cause accuracy issues.' % (self.attrs.get('layer_id', "-1"), str(self.type), tname, requre_bits, actual_bits), log_once=True)
            elif actual_bits > requre_bits:
                OPT_DEBUG('layer_id=%s, layer_type=%s : requred bits for tensor "%s" is %d,'
                          ' but actually got %d, which is due to accuracy consideration.' % (self.attrs.get('layer_id', "-1"), str(self.type), tname, requre_bits, actual_bits), log_once=True)
        if not unquantifiable:
            if 'weights' in self.constants:
                tname = self.constants['weights'].name
                actual_bits = self.constants['weights'].qbits
                requre_bits = self.attrs['q_bits_weight']
                check_qbits_log(tname, requre_bits, actual_bits)
            if 'biases' in self.constants:
                tname = self.constants['biases'].name
                actual_bits = self.constants['biases'].qbits
                requre_bits = self.attrs['q_bits_bias']
                check_qbits_log(tname, requre_bits, actual_bits)
            for o in self.outputs:
                tname = o.name
                actual_bits = o.qbits
                requre_bits = self.attrs['q_bits_activation']
                check_qbits_log(tname, requre_bits, actual_bits)
        return ret

    def param_show(self, *args):

        from AIPUBuilder.Optimizer.logger import OPT_DEBUG
        import torch

        def show_output_min_max_scale(id):
            OPT_DEBUG(
                f'\t{id}. output min/max: [min, max, mean, std, scale, dtype]:')
            for out in self.outputs:
                OPT_DEBUG(f'\t\t{out.name}:')
                OPT_DEBUG(
                    f'\t\t[{out.min: <.6f}, {out.max: <.6f}, {out.running_mean: <.6f}, {out.running_std: <.6f}, {out.scale: <.6f}, {str(out.dtype)}]')

        def show_constant_min_max_scale(id):
            if self.constants:
                OPT_DEBUG(
                    f'\t{id}. constant min/max: [min, max, mean, std, qmin_v, qmax_v, dtype]:')
                for k, c in self.constants.items():
                    qmax_v = torch.max(c.betensor).item()
                    qmin_v = torch.min(c.betensor).item()
                    OPT_DEBUG(f'\t\t{k}:')
                    OPT_DEBUG(
                        f'\t\t[{c.min: <.6f}, {c.max: <.6f}, {c.running_mean: <.6f}, {c.running_std: <.6f}, {qmin_v}, {qmax_v}, {str(c.dtype)}]')

        def show_scale_shift(id):
            from AIPUBuilder.Optimizer.utils import list_any_to_str
            pstr = ''
            for ele in ['scale_value', 'shift_value', 'scale_type', 'shift_type']:
                if ele in self.params:
                    pstr += f'{ele}: {list_any_to_str(self.params[ele])}, '
            if pstr != '':
                pstr = f'{str(id)}. {pstr}'
                OPT_DEBUG(f'\t{pstr}')

        def show_constant_quantize_param(id):
            pstr = ''
            for ele in ['weights', 'biases', 'negative_slope']:
                if ele in self.constants:
                    tensor = self.constants[ele]
                    pstr += f'[{ele}] scale:{str(tensor.scale)}, bits:{str(tensor.qbits)} '
            if pstr != '':
                pstr = f'{str(id)}. {pstr}'
                OPT_DEBUG(f'\t{pstr}')

        param_show_dict = {
            'output_min_max_scale': show_output_min_max_scale,
            'show_constant_quantize_param': show_constant_quantize_param,
            'scale_shift': show_scale_shift,
            'constant_min_max_scale': show_constant_min_max_scale,
        }

        OPT_DEBUG(
            f"layer_id={self.attrs.get('layer_id','-1')}, layer_type={self.type}")
        # OPT_DEBUG('layer_name=%s, layer_type=%s'% (self.name, self.type))
        for idx, v in enumerate(param_show_dict.values()):
            v(idx)


def _get_current_batch_size(self):
    if "current_batch_size" in self.attrs:
        return self.attrs["current_batch_size"]
    elif 'batch_size_in_IR' in self.attrs:
        return self.attrs["batch_size_in_IR"]
    else:
        return 1


def _set_current_batch_size(self, current_batch_size):
    self.attrs["current_batch_size"] = current_batch_size


def _get_current_batch_idx(self):
    if "current_batch_idx" in self.attrs:
        return self.attrs["current_batch_idx"]
    else:
        return 0


def _set_current_batch_idx(self, current_batch_idx):
    self.attrs["current_batch_idx"] = current_batch_idx


def _is_quantized(self):
    return "quantized" in self.attrs and self.attrs["quantized"]


def _set_quantized(self, flag):
    self.attrs["quantized"] = flag


def _is_additional(self):
    return "additional" in self.attrs and self.attrs["additional"]


def _set_additional(self, flag):
    self.attrs["additional"] = flag


def _whether_force_dtype_int(self):
    return "force_dtype_int" in self.attrs and self.attrs["force_dtype_int"]


def _set_force_dtype_int(self, flag):
    self.attrs["force_dtype_int"] = flag


def _whether_force_shift_positive(self):
    return "force_shift_positive" in self.attrs and self.attrs["force_shift_positive"]


def _set_force_shift_positive(self, flag):
    self.attrs["force_shift_positive"] = flag


PyNode.current_batch_size = property(_get_current_batch_size, _set_current_batch_size)
PyNode.current_batch_idx = property(_get_current_batch_idx, _set_current_batch_idx)
PyNode.quantized = property(_is_quantized, _set_quantized)
PyNode.additional = property(_is_additional, _set_additional)
PyNode.force_dtype_int = property(_whether_force_dtype_int, _set_force_dtype_int)
PyNode.force_shift_positive = property(_whether_force_shift_positive, _set_force_shift_positive)


class OpDtypeSpec:
    def __init__(self) -> None:
        self.in_dtypes = []
        self.out_dtypes = []


def get_op_dtype_spec(ntype):
    # return the operator (ntype) 's supported input and output dtype specifics
    from AIPUBuilder.Optimizer.framework.pycore.pytype import Dtype, OpType
    from AIPUBuilder.Optimizer.logger import OPT_WARN
    # fill each op's dtype spec here if necessary
    # if OpType.Abs == ntype :
    #     spec1 = OpDtypeSpec()
    #     spec1.in_dtypes = [Dtype.UINT8,]
    #     spec1.out_dtypes = [Dtype.UINT8,]
    #     return [spec1,]
    try:
        from AIPUBuilder._C._dtypes import get_dtypes
        from AIPUBuilder._C._core import OpType as _OpType
        from AIPUBuilder._C._core import _py_register_optype
        dt_dict = {dt.name: dt for dt in Dtype}

        def _register_optype(t: str):
            if t not in _OpType.__entries:
                v = _OpType(_py_register_optype(t))
                setattr(_OpType, t, _OpType(v))
                _OpType.__entries[t] = (v, None)  # (value, doc string)
            else:
                return _OpType.__entries[t][0]
            return v
        ot_dict = {str(t[0]): t[0] for t in _OpType.__entries.values()}
        ot = ot_dict[ntype] if ntype in ot_dict.keys() else _register_optype(str(ntype)[7:])
        c_specs = get_dtypes(ot)
        dtype_specs = []
        for c_spec in c_specs:
            spec = OpDtypeSpec()
            spec.in_dtypes = []
            spec.out_dtypes = []
            for s in c_spec.in_dtypes:
                spec.in_dtypes.append(dt_dict[str(s.name)])
            for s in c_spec.out_dtypes:
                spec.out_dtypes.append(dt_dict[str(s.name)])
            dtype_specs.append(spec)
        return dtype_specs
    except Exception as e:
        # OPT_WARN(f"the node.type={ntype} has get [] dtype list from lib implementation and give the exception msg: {e}")
        OPT_WARN(f"get_op_dtype_spec() failed to call _C api with the exception msg: {e}", log_once=True)
        OPT_WARN("Will lookup static table which may not be up to date.", log_once=True)
        return lookup_static_op_dtype_spec_table(ntype)


#################################################################################
# print lib's dtype spec for decoupling from _C modules in some cases
# from AIPUBuilder.Optimizer.framework.pycore.pytype import OpType
# from AIPUBuilder.Optimizer.logger import tqdm
# import sys
# all_op_dtype_spec = {}
# with tqdm(total=len(OpType.__dict__.values()), desc='traverse_all_op_dtype_spec', file=sys.stdout, leave=False) as pbar:
#     for ot in OpType.__dict__.values():
#         all_op_dtype_spec[ot] = []
#         for c_spec in get_op_dtype_spec(ot) :
#             in_dtypes = []
#             for dt in c_spec.in_dtypes:
#                 in_dtypes.append(str(dt.name))
#             out_dtypes = []
#             for dt in c_spec.out_dtypes:
#                 out_dtypes.append(str(dt.name))
#             all_op_dtype_spec[ot].append((in_dtypes, out_dtypes))
#         pbar.update(1)
#     pbar.refresh()
# print('---------------------------------')
# print(all_op_dtype_spec)
# print('---------------------------------')
#################################################################################
g_all_op_dtype_spec = {'OpType.Abs': [(['INT8'], ['UINT8']), (['UINT8'], ['UINT8']), (['INT16'], ['UINT16']), (['UINT16'], ['UINT16'])], 'OpType.AccidentalHits': [], 'OpType.Acos': [], 'OpType.Acosh': [], 'OpType.Activation': [(['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['INT8'], ['INT16']), (['INT8'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8']), (['UINT8'], ['INT16']), (['UINT8'], ['UINT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT16'], ['INT8']), (['INT16'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT16'], ['INT8']), (['UINT16'], ['UINT8'])], 'OpType.Add': [(['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT8']), (['INT8', 'INT8'], ['INT16']), (['INT8', 'INT8'], ['UINT16']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT16']), (['UINT8', 'UINT8'], ['UINT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['UINT16']), (['INT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT8']), (['UINT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT16'], ['INT8']), (['UINT16', 'UINT16'], ['UINT8']), (['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT16']), (['INT8', 'UINT8'], ['UINT16']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'INT8'], ['INT16']), (['UINT8', 'INT8'], ['UINT16']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT16'], ['UINT16']), (['INT16', 'UINT16'], ['INT8']), (['INT16', 'UINT16'], ['UINT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['UINT8'])], 'OpType.ArgMinMax': [], 'OpType.Asin': [], 'OpType.Asinh': [], 'OpType.BNLL': [(['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.BasicLSTM': [(['INT8', 'INT8', 'INT8'], ['INT8', 'INT8', 'INT8']), (['INT8', 'INT8', 'INT8'], ['INT8', 'INT8']), (['INT8', 'INT8', 'INT8'], ['INT8']), (['UINT8', 'INT8', 'INT8'], ['INT8', 'INT8', 'INT8']), (['UINT8', 'INT8', 'INT8'], ['INT8', 'INT8']), (['UINT8', 'INT8', 'INT8'], ['INT8']), (['INT16', 'INT16', 'INT16'], ['INT16', 'INT16', 'INT16']), (['INT16', 'INT16', 'INT16'], ['INT16', 'INT16']), (['INT16', 'INT16', 'INT16'], ['INT16']), (['UINT16', 'INT16', 'INT16'], ['INT16', 'INT16', 'INT16']), (['UINT16', 'INT16', 'INT16'], ['INT16', 'INT16']), (['UINT16', 'INT16', 'INT16'], ['INT16'])], 'OpType.BatchNorm': [(['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['INT8'], ['INT16']), (['INT8'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8']), (['UINT8'], ['INT16']), (['UINT8'], ['UINT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT16'], ['INT8']), (['INT16'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT16'], ['INT8']), (['UINT16'], ['UINT8'])], 'OpType.BatchToDepth': [(['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.BatchToSpace': [(['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['FP16'], ['FP16'])], 'OpType.BiasAdd': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT32'], ['INT32']), (['UINT32'], ['UINT32'])], 'OpType.BitShift': [(['UINT8', 'UINT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT8']), (['UINT16', 'UINT8'], ['UINT16']), (['INT16', 'UINT8'], ['INT16']), (['UINT32', 'UINT8'], ['UINT32']), (['INT32', 'UINT8'], ['INT32'])], 'OpType.BoundingBox': [(['INT16', 'INT8'], ['INT16', 'UINT16', 'UINT16'])], 'OpType.CELU': [], 'OpType.CRELU': [], 'OpType.CTCBeamDecoder': [], 'OpType.CTCGreedyDecoder': [(['INT8', 'UINT8'], ['UINT16']), (['INT8', 'UINT16'], ['UINT16']), (['UINT8', 'UINT8'], ['UINT16']), (['UINT8', 'UINT16'], ['UINT16'])], 'OpType.Cast': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT32'], ['INT32']), (['UINT32'], ['UINT32']), (['INT8'], ['UINT8']), (['UINT8'], ['INT8']), (['INT16'], ['UINT16']), (['UINT16'], ['INT16']), (['INT32'], ['UINT32']), (['UINT32'], ['INT32']), (['INT8'], ['INT16']), (['UINT8'], ['UINT16']), (['INT8'], ['UINT16']), (['UINT8'], ['INT16']), (['INT8'], ['INT32']), (['UINT8'], ['UINT32']), (['INT8'], ['UINT32']), (['UINT8'], ['INT32']), (['INT16'], ['INT8']), (['UINT16'], ['UINT8']), (['INT16'], ['UINT8']), (['UINT16'], ['INT8']), (['INT16'], ['INT32']), (['UINT16'], ['UINT32']), (['INT16'], ['UINT32']), (['UINT16'], ['INT32']), (['INT32'], ['INT16']), (['UINT32'], ['UINT16']), (['INT32'], ['UINT16']), (['UINT32'], ['INT16']), (['INT32'], ['INT8']), (['UINT32'], ['UINT8']), (['INT32'], ['UINT8']), (['UINT32'], ['INT8']), (['INT8'], ['FP16']), (['INT16'], ['FP16']), (['INT32'], ['FP16']), (['UINT8'], ['FP16']), (['UINT16'], ['FP16']), (['UINT32'], ['FP16']), (['INT8'], ['FP32']), (['INT16'], ['FP32']), (['INT32'], ['FP32']), (['UINT8'], ['FP32']), (['UINT16'], ['FP32']), (['UINT32'], ['FP32']), (['FP16'], ['INT8']), (['FP16'], ['INT16']), (['FP16'], ['INT32']), (['FP16'], ['UINT8']), (['FP16'], ['UINT16']), (['FP16'], ['UINT32']), (['FP32'], ['INT8']), (['FP32'], ['INT16']), (['FP32'], ['INT32']), (['FP32'], ['UINT8']), (['FP32'], ['UINT16']), (['FP32'], ['UINT32'])], 'OpType.Ceil': [(['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.ChannelShuffle': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['FP16'], ['FP16'])], 'OpType.Clip': [], 'OpType.Compress': [], 'OpType.Concat': [], 'OpType.Constant': [([], ['INT8']), ([], ['UINT8']), ([], ['INT16']), ([], ['UINT16']), ([], ['INT32']), ([], ['UINT32'])], 'OpType.ConvInteger': [], 'OpType.ConvTranspose': [(['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16'])], 'OpType.ConvTranspose3D': [(['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16'])], 'OpType.Convolution': [(['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16'])], 'OpType.Convolution3D': [(['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16'])], 'OpType.Cosh': [], 'OpType.Cosine': [], 'OpType.Count': [], 'OpType.Crop': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['FP16'], ['FP16'])], 'OpType.CropAndResize': [(['INT8', 'UINT16', 'UINT8'], ['INT8']), (['UINT8', 'UINT16', 'UINT8'], ['UINT8']), (['INT16', 'UINT16', 'UINT8'], ['INT16']), (['INT8', 'UINT16', 'UINT8'], ['UINT8'])], 'OpType.DataStride': [], 'OpType.DeQuantize': [(['INT16'], ['FP16']), (['INT16'], ['BFP16']), (['UINT16'], ['FP16']), (['UINT16'], ['BFP16'])], 'OpType.DecodeBox': [(['UINT8', 'INT8'], ['INT16', 'UINT16', 'UINT16', 'UINT8', 'UINT16'])], 'OpType.DepthToSpace': [(['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.DepthwiseConv': [], 'OpType.DetectionOutput': [(['UINT8', 'INT8', 'INT16'], ['UINT8', 'INT16', 'UINT16', 'UINT16', 'UINT16']), (['UINT8', 'INT8', 'UINT16'], ['UINT8', 'INT16', 'UINT16', 'UINT16', 'UINT16'])], 'OpType.Div': [(['INT8', 'INT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT8']), (['UINT8', 'INT8'], ['INT8']), (['FP16'], ['FP16'])], 'OpType.ELU': [], 'OpType.Eltwise': [(['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT8']), (['INT8', 'INT8'], ['INT16']), (['INT8', 'INT8'], ['UINT16']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT16']), (['UINT8', 'UINT8'], ['UINT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['UINT16']), (['INT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT8']), (['UINT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT16'], ['INT8']), (['UINT16', 'UINT16'], ['UINT8']), (['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT16']), (['INT8', 'UINT8'], ['UINT16']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'INT8'], ['INT16']), (['UINT8', 'INT8'], ['UINT16']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT16'], ['UINT16']), (['INT16', 'UINT16'], ['INT8']), (['INT16', 'UINT16'], ['UINT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['UINT8'])], 'OpType.Erf': [(['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Exp': [], 'OpType.FakeQuantWithMinMaxVars': [], 'OpType.Filter': [], 'OpType.Floor': [(['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.FractionalPool': [(['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.FullyConnected': [(['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16'])], 'OpType.GELU': [], 'OpType.GRUv1': [(['INT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['INT8']), (['INT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['INT16'])], 'OpType.GRUv3': [(['INT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['INT8']), (['INT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['INT16'])], 'OpType.Gather': [(['INT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['UINT16']), (['INT8', 'INT16'], ['INT8']), (['UINT8', 'INT16'], ['UINT8']), (['INT32', 'INT16'], ['INT32']), (['UINT32', 'INT16'], ['UINT32']), (['INT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['UINT16']), (['INT8', 'UINT16'], ['INT8']), (['UINT8', 'UINT16'], ['UINT8']), (['INT32', 'UINT16'], ['INT32']), (['UINT32', 'UINT16'], ['UINT32']), (['INT16', 'INT8'], ['INT16']), (['UINT16', 'INT8'], ['UINT16']), (['INT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT8']), (['INT32', 'INT8'], ['INT32']), (['UINT32', 'INT8'], ['UINT32']), (['INT16', 'UINT8'], ['INT16']), (['UINT16', 'UINT8'], ['UINT16']), (['INT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8']), (['INT32', 'UINT8'], ['INT32']), (['UINT32', 'UINT8'], ['UINT32']), (['INT16', 'INT32'], ['INT16']), (['UINT16', 'INT32'], ['UINT16']), (['INT8', 'INT32'], ['INT8']), (['UINT8', 'INT32'], ['UINT8']), (['INT32', 'INT32'], ['INT32']), (['UINT32', 'INT32'], ['UINT32']), (['INT16', 'UINT32'], ['INT16']), (['UINT16', 'UINT32'], ['UINT16']), (['INT8', 'UINT32'], ['INT8']), (['UINT8', 'UINT32'], ['UINT8']), (['INT32', 'UINT32'], ['INT32']), (['UINT32', 'UINT32'], ['UINT32']), (['FP16', 'UINT32'], ['FP16']), (['FP16', 'INT32'], ['FP16']), (['FP16', 'UINT16'], ['FP16']), (['FP16', 'INT16'], ['FP16']), (['FP16', 'UINT8'], ['FP16']), (['FP16', 'INT8'], ['FP16'])], 'OpType.GatherElements': [], 'OpType.GatherND': [(['INT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['UINT16']), (['INT8', 'INT16'], ['INT8']), (['UINT8', 'INT16'], ['UINT8']), (['INT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['UINT16']), (['INT8', 'UINT16'], ['INT8']), (['UINT8', 'UINT16'], ['UINT8']), (['INT16', 'INT8'], ['INT16']), (['UINT16', 'INT8'], ['UINT16']), (['INT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT8']), (['INT16', 'UINT8'], ['INT16']), (['UINT16', 'UINT8'], ['UINT16']), (['INT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8']), (['INT16', 'INT32'], ['INT16']), (['UINT16', 'INT32'], ['UINT16']), (['INT8', 'INT32'], ['INT8']), (['UINT8', 'INT32'], ['UINT8']), (['INT16', 'UINT32'], ['INT16']), (['UINT16', 'UINT32'], ['UINT16']), (['INT8', 'UINT32'], ['INT8']), (['UINT8', 'UINT32'], ['UINT8'])], 'OpType.Gemm': [], 'OpType.GenerateProposals': [(['UINT8', 'INT8', 'UINT16'], ['UINT8', 'UINT16', 'UINT16', 'UINT16']), (['UINT8', 'INT8', 'UINT8'], ['UINT8', 'UINT16', 'UINT16', 'UINT16'])], 'OpType.GridSample': [(['INT8', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['INT16'])], 'OpType.GroupNorm': [], 'OpType.HardSigmoid': [], 'OpType.Hardmax': [], 'OpType.Hardswish': [], 'OpType.InTopK': [(['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.Input': [], 'OpType.InstanceNorm': [(['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Interp': [], 'OpType.LRN': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16'])], 'OpType.LayerNorm': [(['INT16'], ['INT16']), (['INT16'], [
    'UINT16']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.LeakyRELU': [], 'OpType.Log': [], 'OpType.LogSoftmax': [(['INT16'], ['INT16']), (['UINT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT8'], ['INT8'])], 'OpType.Logical': [(['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT8'], ['UINT8']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8']), (['INT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT8']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['UINT16']), (['INT16', 'UINT16'], ['INT8']), (['INT16', 'UINT16'], ['UINT8']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT16'], ['UINT16']), (['UINT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['UINT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'UINT16'], ['INT8']), (['UINT16', 'UINT16'], ['UINT8']), (['UINT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['UINT16'])], 'OpType.MISH': [], 'OpType.MVN': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['FP16'], ['FP16'])], 'OpType.MatMul': [(['INT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT8']), (['UINT8', 'INT8'], ['UINT8']), (['INT8', 'INT8'], ['INT16']), (['UINT8', 'INT8'], ['INT16']), (['INT8', 'INT8'], ['UINT16']), (['UINT8', 'INT8'], ['UINT16']), (['INT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['UINT16']), (['UINT16', 'INT16'], ['UINT16']), (['INT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT8']), (['UINT16', 'INT16'], ['UINT8'])], 'OpType.MatMulInteger': [], 'OpType.MaxPoolingWithArgMax': [(['INT8'], ['INT8', 'INT32'])], 'OpType.MaxRoiPool': [(['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT16'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8']), (['UINT8', 'UINT16'], ['UINT8']), (['INT16', 'UINT8'], ['INT16']), (['INT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT8'], ['UINT16']), (['UINT16', 'UINT16'], ['UINT16'])], 'OpType.MaxUnpool': [(['INT8', 'INT32'], ['INT8']), (['UINT8', 'INT32'], ['UINT8']), (['INT16', 'INT32'], ['INT16']), (['UINT16', 'INT32'], ['UINT16'])], 'OpType.Meshgrid': [], 'OpType.Mod': [(['UINT8', 'UINT8'], ['UINT8']), (['INT8', 'INT8'], ['INT8'])], 'OpType.Moments': [], 'OpType.Mul': [(['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT8']), (['INT8', 'INT8'], ['INT16']), (['INT8', 'INT8'], ['UINT16']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT16']), (['UINT8', 'UINT8'], ['UINT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['UINT16']), (['INT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT8']), (['UINT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT16'], ['INT8']), (['UINT16', 'UINT16'], ['UINT8']), (['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT16']), (['INT8', 'UINT8'], ['UINT16']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'INT8'], ['INT16']), (['UINT8', 'INT8'], ['UINT16']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT16'], ['UINT16']), (['INT16', 'UINT16'], ['INT8']), (['INT16', 'UINT16'], ['UINT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['UINT8'])], 'OpType.NMS': [(['INT16', 'UINT16', 'INT16', 'UINT8'], ['INT16', 'UINT16', 'UINT8', 'UINT16']), (['INT16', 'UINT16', 'UINT16', 'UINT8'], ['INT16', 'UINT16', 'UINT8', 'UINT16']), (['INT16', 'UINT16', 'INT16', 'UINT16'], ['INT16', 'UINT16', 'UINT16', 'UINT16']), (['INT16', 'UINT16', 'UINT16', 'UINT16'], ['INT16', 'UINT16', 'UINT16', 'UINT16'])], 'OpType.Negative': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16'])], 'OpType.Normalization': [(['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.OneHot': [(['INT8'], ['UINT8']), (['INT8'], ['INT8']), (['INT8'], ['UINT16']), (['INT8'], ['INT16']), (['INT8'], ['UINT32']), (['INT8'], ['INT32']), (['UINT8'], ['UINT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT16']), (['UINT8'], ['INT16']), (['UINT8'], ['UINT32']), (['UINT8'], ['INT32']), (['INT16'], ['UINT8']), (['INT16'], ['INT8']), (['INT16'], ['UINT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT32']), (['INT16'], ['INT32']), (['UINT16'], ['UINT8']), (['UINT16'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT32']), (['UINT16'], ['INT32']), (['INT32'], ['UINT8']), (['INT32'], ['INT8']), (['INT32'], ['UINT16']), (['INT32'], ['INT16']), (['INT32'], ['UINT32']), (['INT32'], ['INT32']), (['UINT32'], ['UINT8']), (['UINT32'], ['INT8']), (['UINT32'], ['UINT16']), (['UINT32'], ['INT16']), (['UINT32'], ['UINT32']), (['UINT32'], ['INT32'])], 'OpType.OverlapAdd': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16'])], 'OpType.PRELU': [], 'OpType.Pad': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16'])], 'OpType.Permute': [], 'OpType.Pooling': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['UINT8']), (['INT16'], ['UINT16'])], 'OpType.Pooling3D': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16'])], 'OpType.PostNMS1': [(['INT16', 'UINT16', 'UINT16'], ['INT16', 'UINT16'])], 'OpType.PostNMS2': [], 'OpType.Pow': [(['INT8', 'INT8'], ['INT8']), (['INT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8']), (['UINT8', 'INT8'], ['UINT8'])], 'OpType.Proposal': [(['UINT8', 'INT8'], ['UINT8', 'INT16', 'UINT16', 'UINT16'])], 'OpType.PyramidROIAlign': [(['INT16', 'INT8', 'INT8', 'INT8', 'INT8'], ['INT8']), (['UINT16', 'INT8', 'INT8', 'INT8', 'INT8'], ['INT8']), (['INT16', 'UINT8', 'UINT8', 'UINT8', 'UINT8'], ['UINT8']), (['UINT16', 'UINT8', 'UINT8', 'UINT8', 'UINT8'], ['UINT8'])], 'OpType.Quantize': [(['FP16'], ['INT16']), (['BFP16'], ['INT16']), (['FP16'], ['UINT16']), (['BFP16'], ['UINT16'])], 'OpType.RELU': [], 'OpType.RELU6': [], 'OpType.ROIPooling': [], 'OpType.Reciprocal': [], 'OpType.Reduce': [(['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT8'], ['UINT8'])], 'OpType.Region': [(['INT8'], ['UINT8', 'INT16', 'INT16', 'INT16', 'INT16'])], 'OpType.RegionFuse': [], 'OpType.Repeat': [], 'OpType.Reshape': [], 'OpType.Resize': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16'])], 'OpType.ReverseSequence': [(['INT8', 'UINT16'], ['INT8']), (['UINT8', 'UINT16'], ['UINT8']), (['INT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['UINT16'])], 'OpType.RgbToYuv': [], 'OpType.RNN': [], 'OpType.RoiAlign': [(['INT16', 'UINT16'], ['INT16']), (['INT8', 'UINT16'], ['INT8']), (['UINT8', 'UINT16'], ['UINT8']), (['INT16', 'UINT8'], ['INT16']), (['INT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.Round': [], 'OpType.Rsqrt': [], 'OpType.SELU': [], 'OpType.SHRINK': [], 'OpType.ScatterElements': [(['INT8', 'INT32', 'INT8'], ['INT8']), (['UINT8', 'INT32', 'UINT8'], ['UINT8']), (['INT16', 'INT32', 'INT16'], ['INT16']), (['UINT16', 'INT32', 'UINT16'], ['UINT16']), (['INT8', 'INT16', 'INT8'], ['INT8']), (['UINT8', 'INT16', 'UINT8'], ['UINT8']), (['INT16', 'INT16', 'INT16'], ['INT16']), (['UINT16', 'INT16', 'UINT16'], ['UINT16']), (['INT8', 'INT8', 'INT8'], ['INT8']), (['UINT8', 'INT8', 'UINT8'], ['UINT8']), (['INT16', 'INT8', 'INT16'], ['INT16']), (['UINT16', 'INT8', 'UINT16'], ['UINT16'])], 'OpType.ScatterND': [(['INT16', 'UINT16', 'INT16'], ['INT16']), (['INT8', 'UINT16', 'INT8'], ['INT8']), (['UINT8', 'UINT16', 'UINT8'], ['UINT8']), (['INT16', 'UINT8', 'INT16'], ['INT16']), (['INT8', 'UINT8', 'INT8'], ['INT8']), (['UINT8', 'UINT8', 'UINT8'], ['UINT8'])], 'OpType.SegmentReduce': [(['INT8', 'UINT16'], ['INT8'])], 'OpType.Sigmoid': [], 'OpType.Sign': [(['INT8'], ['UINT8']), (['INT16'], ['UINT16'])], 'OpType.Silu': [], 'OpType.Sine': [], 'OpType.Sinh': [], 'OpType.Slice': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['FP16'], ['FP16'])], 'OpType.Softmax': [(['INT16'], ['UINT16']), (['UINT16'], ['UINT16']), (['INT16'], ['INT16']), (['UINT16'], ['INT16']), (['INT8'], ['UINT8']), (['UINT8'], ['UINT8']), (['INT8'], ['INT8']), (['UINT8'], ['INT8']), (['FP16'], ['FP16'])], 'OpType.Softplus': [], 'OpType.Softsign': [], 'OpType.Sort': [], 'OpType.SpaceToBatch': [(['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.SpaceToDepth': [(['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Split': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['FP16'], ['FP16'])], 'OpType.Sqrt': [], 'OpType.Square': [], 'OpType.SquaredDifference': [(['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT8']), (['INT8', 'INT8'], ['INT16']), (['INT8', 'INT8'], ['UINT16']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT16']), (['UINT8', 'UINT8'], ['UINT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['UINT16']), (['INT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT8']), (['UINT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT16'], ['INT8']), (['UINT16', 'UINT16'], ['UINT8']), (['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT16']), (['INT8', 'UINT8'], ['UINT16']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'INT8'], ['INT16']), (['UINT8', 'INT8'], ['UINT16']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT16'], ['UINT16']), (['INT16', 'UINT16'], ['INT8']), (['INT16', 'UINT16'], ['UINT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['UINT8'])], 'OpType.Squeeze': [], 'OpType.StridedSlice': [], 'OpType.Sub': [(['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT8']), (['INT8', 'INT8'], ['INT16']), (['INT8', 'INT8'], ['UINT16']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT16']), (['UINT8', 'UINT8'], ['UINT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['UINT16']), (['INT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT8']), (['UINT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT16'], ['INT8']), (['UINT16', 'UINT16'], ['UINT8']), (['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT16']), (['INT8', 'UINT8'], ['UINT16']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'INT8'], ['INT16']), (['UINT8', 'INT8'], ['UINT16']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT16'], ['UINT16']), (['INT16', 'UINT16'], ['INT8']), (['INT16', 'UINT16'], ['UINT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['UINT8'])], 'OpType.THRESHOLDEDRELU': [], 'OpType.Tan': [], 'OpType.Tanh': [], 'OpType.Tile': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT32'], ['INT32']), (['UINT32'], ['UINT32'])], 'OpType.TopK': [], 'OpType.Transpose': [(['INT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['FP16'], ['FP16']), (['INT32'], ['INT32']), (['UINT32'], ['UINT32'])], 'OpType.UpsampleByIndex': [(['INT8', 'INT32'], ['INT8']), (['UINT8', 'INT32'], ['UINT8']), (['INT16', 'INT32'], ['INT16']), (['UINT16', 'INT32'], ['UINT16'])], 'OpType.Where': [(['INT8', 'INT8', 'INT8'], ['INT8']), (['UINT8', 'UINT8', 'UINT8'], ['UINT8']), (['INT16', 'INT16', 'INT16'], ['INT16']), (['UINT16', 'UINT16', 'UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['UINT8'], ['UINT8']), (['INT16'], ['INT16']), (['UINT16'], ['UINT16'])], 'OpType.YuvToRgb': [], 'OpType.ZeroFraction': [], 'OpType.Swish': [], 'OpType.Bitwise': [], 'OpType.Trunc': [], 'OpType.Cumulate': [(['INT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.MultiboxTransformLoc': [(['UINT8', 'INT8', 'INT16'], ['INT16', 'UINT16'])], 'OpType.Dilation': [(['INT16'], ['INT16']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['INT8'], ['INT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Erosion': [(['INT16'], ['INT16']), (['UINT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT8'], ['INT8'])], 'OpType.GetValidCount': [], 'OpType.NormalizedMoments': [(['INT8', 'INT8', 'INT8'], ['INT8', 'INT8']), (['INT8', 'INT8', 'UINT8'], ['INT8', 'INT8']), (['UINT8', 'INT8', 'INT8'], ['INT8', 'INT8']), (['UINT8', 'INT8', 'UINT8'], ['UINT8', 'INT8']), (['INT16', 'INT16', 'INT16'], ['INT16', 'INT16']), (['INT16', 'INT16', 'UINT16'], ['INT16', 'INT16']), (['UINT16', 'INT16', 'INT16'], ['INT16', 'INT16']), (['UINT16', 'INT16', 'UINT16'], ['UINT16', 'INT16'])], 'OpType.EmbeddingLookupSparse': []}


def lookup_static_op_dtype_spec_table(ntype):
    if str(ntype) in g_all_op_dtype_spec.keys():
        from AIPUBuilder.Optimizer.utils.dtype_utils import str2dtype
        dtype_specs = []
        for c_spec in g_all_op_dtype_spec[str(ntype)]:
            spec = OpDtypeSpec()
            spec.in_dtypes = []
            spec.out_dtypes = []
            for s in c_spec[0]:
                spec.in_dtypes.append(str2dtype(s))
            for s in c_spec[1]:
                spec.out_dtypes.append(str2dtype(s))
            dtype_specs.append(spec)
        return dtype_specs
    else:
        return []
#################################################################################
