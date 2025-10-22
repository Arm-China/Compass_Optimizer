# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3

__all__ = [
    "PyNode",
]


class ParamDict(dict):
    def clone(self):
        # IR fields, need deepcopy
        import copy
        d = self.__class__()
        for k, v in self.items():
            d[k] = copy.deepcopy(v)
        return d


class AttrDict(dict):
    # config for dev usage, needn't deepcopy as may cause recursive clone when node object exist in attrs
    def clone(self):
        d = self.__class__()
        for k, v in self.items():
            if isinstance(v, list):
                d[k] = [e for e in v]
            elif isinstance(v, dict):
                d[k] = {ek: ev for ek, ev in v.items()}
            else:
                d[k] = v
        return d


class OpDtypeSpec:
    def __init__(self) -> None:
        self.in_dtypes = []
        self.out_dtypes = []

    def __repr__(self):
        return f"{self.in_dtypes} -> {self.out_dtypes}"


class PyNode:
    # __slots__ = ('name', 'type', 'params', 'attrs', 'constants', 'inputs',
    #              'outputs', 'parents', 'children', 'placeholders', 'graph')
    all_op_dtype_spec = {}

    def __init__(self, name, type) -> None:
        from AIPUBuilder.Optimizer.framework.pycore.pytype import OpTypeValue
        self.name = str(name)
        self.type = OpTypeValue(str(type))
        self.params = ParamDict()
        self.attrs = AttrDict()
        self.constants = {}  # weights, biases
        self.inputs = ()
        self.outputs = ()
        self.parents = ()
        self.children = ()
        self.placeholders = []  # internal Tensors
        self.graph = None  # the belonged graph
        self.forward_hook = lambda node: True  # will be called as 'forward_hook(node)' every time after node.forward()

    def clone(self, name=None):
        import copy
        if name is None:
            name = self.name + '_clone' if not self.name.endswith("_clone") else self.name
        n = self.__class__(name, self.type)
        for k, v in self.params.items():
            n.params[k] = copy.deepcopy(v)
        n.attrs.update(self.attrs.clone())
        for k, v in self.constants.items():
            n.constants[k] = v.clone(v.name)
        for t in self.inputs:
            n.add_input(t.clone(t.name))
        for t in self.outputs:
            n.add_output(t.clone(t.name))
        for t in self.placeholders:
            n.placeholders.append(t.clone(t.name))
        n.graph = self.graph
        return n

    def get_sub_graph(self, gname: str):
        from AIPUBuilder.Optimizer.framework.pycore.pygraph import PyGraph
        sub_g_name = self.params[gname]
        if len(self.graph.subgraph_map) >= 1:
            sub_g = self.graph.subgraph_map[sub_g_name]
        else:
            # nested cases
            sub_g = self.graph.root_graph.subgraph_map[sub_g_name]
        return sub_g

    def clear_inputs(self):
        self.inputs = ()

    def clear_outputs(self):
        self.outputs = ()

    def add_input(self, t, idx=-1):
        k = idx if idx >= 0 else len(self.inputs) + 1 + idx
        k = max(0, min(k, len(self.inputs)))
        self.inputs = self.inputs[:k] + (t, ) + self.inputs[k:]
        if self.graph:
            # update edges' relationship in corresponding graph
            self.graph.init_networkx()

    def add_output(self, t, idx=-1):
        t.is_act = True
        k = idx if idx >= 0 else len(self.outputs) + 1 + idx
        k = max(0, min(k, len(self.outputs)))
        self.outputs = self.outputs[:k] + (t, ) + self.outputs[k:]
        if self.graph:
            # update edges' relationship in corresponding graph
            self.graph.init_networkx()

    def remove_input(self, t_or_idx):
        from AIPUBuilder.Optimizer.framework import PyTensor
        from AIPUBuilder.Optimizer.logger import OPT_WARN, OPT_ERROR
        flag = False
        if isinstance(t_or_idx, int):
            idx = t_or_idx + len(self.inputs) if t_or_idx < 0 else t_or_idx
            flag = True
            if t_or_idx > len(self.inputs) - 1:
                OPT_ERROR(f"{self} remove_input idx(={t_or_idx})  exceeds the input length.")
                flag = False
        elif isinstance(t_or_idx, PyTensor):
            idx, flag = (self.inputs.index(t_or_idx), True) if t_or_idx in self.inputs else (None, False)
        else:
            OPT_ERROR(
                f"when remove_input, the input args[1] type is {type(t_or_idx)}, which now only support PyTensor or int")

        if flag:
            self.inputs = self.inputs[: idx] + self.inputs[idx+1:]
            if self.graph:
                # update edges' relationship in corresponding graph
                self.graph.init_networkx()
        return idx

    def remove_output(self, t_or_idx):
        from AIPUBuilder.Optimizer.framework import PyTensor
        from AIPUBuilder.Optimizer.logger import OPT_WARN, OPT_ERROR
        flag = False
        if isinstance(t_or_idx, int):
            idx = t_or_idx + len(self.outputs) if t_or_idx < 0 else t_or_idx
            flag = True
            if idx > len(self.outputs) - 1:
                OPT_ERROR(f"{self} remove_output idx(={t_or_idx}) exceeds the output length.")
                flag = False
        elif isinstance(t_or_idx, PyTensor):
            idx, flag = (self.outputs.index(t_or_idx), True) if t_or_idx in self.outputs else (None, False)
        else:
            OPT_ERROR(
                f"when remove_output, the input args[1] type is {type(t_or_idx)}, which now only support PyTensor or int")

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

    def get_ancestors(self):
        from AIPUBuilder.Optimizer.framework.pycore.pytype import OpType
        from queue import Queue
        q = Queue(maxsize=0)
        # total parent path node
        ancestors = []
        visited = {}

        def traverse_parents(node):
            count_root = 0
            count_constant = 0
            if node:
                q.put(node)
                ancestors.append(node)
                visited[node.name] = True

                while(q.qsize()):
                    current_node = q.get()
                    # get count_root and count_constant by the way
                    if len(current_node.parents) < 1:
                        count_root += 1
                    if current_node.type == OpType.Constant:
                        count_constant += 1
                    for nparent in current_node.parents:
                        if nparent.name not in visited:
                            visited[nparent.name] = True
                            q.put(nparent)
                            ancestors.insert(0, nparent)
            return count_root, count_constant
        count_root, count_constant = traverse_parents(self)
        return ancestors, count_root, count_constant

    def get_descendants(self):
        from queue import Queue
        q = Queue(maxsize=0)
        # total parent path node
        descendants = []
        visited = {}

        def traverse_children(node):
            count_leafs = 0
            if node:
                q.put(node)
                descendants.append(node)
                visited[node.name] = True

                while(q.qsize()):
                    current_node = q.get()
                    # get count_root and count_constant by the way
                    if len(current_node.children) < 1:
                        count_leafs += 1
                    for nchild in current_node.children:
                        if nchild.name not in visited:
                            visited[nchild.name] = True
                            q.put(nchild)
                            descendants.insert(0, nchild)
            return count_leafs
        count_leafs = traverse_children(self)
        return descendants, count_leafs

    def statistic(self, *, on_constants: bool = True, on_activations: bool = True, reset: bool = True):
        from AIPUBuilder.Optimizer.config import CalibrationStrategyField
        import re
        node_attrs = self.attrs
        dv = ((float('-inf'), float('inf')), '')
        tcmd = [x for x in re.split(
            r',|\(|\)', node_attrs['trim_infinity_before_statistic'].strip()) if x.lower().strip()]
        trim_inf = dv if len(tcmd) < 3 else ((float(tcmd[1]), float(tcmd[2])), str(tcmd[0]))
        if on_constants:
            for _, t in self.constants.items():
                qstrategy = node_attrs['q_strategy_weight']
                r = CalibrationStrategyField._need_statistic_info(qstrategy)
                histc_bins = None if not r['histc'] else node_attrs["histc_bins"]
                statistic_std_mean = r['std_mean']
                t.statistic(running_statistic_momentum=1.0, key_axis=t.key_axis, key_axis_g=t.key_axis_g,
                            histc_bins=histc_bins, statistic_std_mean=statistic_std_mean,
                            trim_infinity=trim_inf,
                            reset=True)
        if on_activations:
            for t in (list(self.outputs) + list(self.placeholders)):
                qstrategy = node_attrs['q_strategy_activation']
                r = CalibrationStrategyField._need_statistic_info(qstrategy)
                histc_bins = None if not r['histc'] else node_attrs["histc_bins"]
                statistic_std_mean = r['std_mean']
                running_statistic_momentum = node_attrs["running_statistic_momentum"]
                t.statistic(running_statistic_momentum=running_statistic_momentum, key_axis=t.key_axis, key_axis_g=t.key_axis_g,
                            histc_bins=histc_bins, statistic_std_mean=statistic_std_mean,
                            trim_infinity=trim_inf,
                            reset=reset)

    def calibration(self, *, on_constants: bool = True, on_activations: bool = True):
        from AIPUBuilder.Optimizer.features.calibration import apply_calibration_strategy
        if on_constants:
            for _, t in self.constants.items():
                qstrategy = self.attrs['q_strategy_weight']
                qmethod = self.attrs['q_mode_weight']
                apply_calibration_strategy(t, qstrategy, qmethod)
        if on_activations:
            for t in (list(self.outputs) + list(self.placeholders)):
                qstrategy = self.attrs['q_strategy_activation']
                qmethod = self.attrs['q_mode_activation']
                apply_calibration_strategy(t, qstrategy, qmethod)

    def forward(self, *args):
        from AIPUBuilder.Optimizer.framework import OP_DICT, Dtype
        from AIPUBuilder.Optimizer.framework.pycore.pytensor import PyTensor
        from AIPUBuilder.Optimizer.utils.quant_tool_utils import linear_dequantize, linear_quantize_clip, linear_dequantize_for_gptq_w4afp8
        from AIPUBuilder.Optimizer.utils.dtype_utils import is_float
        from AIPUBuilder.Optimizer.logger import OPT_WARN, OPT_ERROR, OPT_FATAL, OPT_DEBUG
        from AIPUBuilder.Optimizer.framework.pycore.pytype import OpType
        import torch
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
                    if k in self.constants.keys():
                        constants_betensor_backup[k] = self.constants[k].betensor
                        self.constants[k].betensor = t
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
        maintained_constants_betensor = {}
        # for cases with explicit IR field
        # if "weights" in self.constants.keys() and self.get_param("approximate_method", optional=True, default_value='none').lower() in ['weight_only_quantization', ]:
        # for cases without explicit IR field, currently lib can detect weight only quantization through IR Dtype info
        # if not self.quantized and "weights" in self.constants.keys() and self.get_attrs('weight_only_quantization', optional=True, default_value=False) :
        if not self.quantized and self.type == OpType.FullyConnected and "weights" in self.constants.keys() and (not is_float(self.constants["weights"].dtype)):
            wt = self.constants["weights"]
            if (wt.scale == torch.ones_like(wt.scale)).all() and (wt.zerop == torch.zeros_like(wt.zerop)).all():
                # when wt.qinvariant is false, but scale == 1.0, zerop == 0; better to skip the dtype convertion caused by dequantize
                pass
            else:
                scales = wt.scale
                zps = wt.zerop
                ir_shape_len = len(wt.ir_shape)
                if ir_shape_len and ir_shape_len - 1 >= wt.key_axis:
                    channel_num = wt.ir_shape[wt.key_axis]
                    if scales.numel() == zps.numel() and scales.numel() > 1 and scales.numel() > channel_num:
                        ic = self.inputs[0].ir_shape[-1]
                        group_size = ic // (scales.numel() // channel_num)
                        wt.block_size = group_size
                # dequantize quantized weights
                maintained_constants_betensor['weights'] = wt.betensor
                maintained_constants_betensor['weights_dtype'] = wt.dtype
                wt.betensor = linear_dequantize(wt.betensor, wt.broadcast_scale, wt.broadcast_zerop)
                wt.dtype = Dtype.FP32

        # fit constants dtype
        if self.fit_dtype_enabled and not all(sname in self.constants.keys() for sname in ['scale0', 'scale1']):
            for k, v in self.constants.items():
                v.fit_dtype()
        ret = OP_DICT[self.type](self, *args)
        for ii, iinp in enumerate(self.inputs):
            iinp.betensor = maintained_inp_betensors[ii]
        for kk, vv in maintained_constants_betensor.items():
            '''
            when weight_only_quantization, we save weight.dtype in the maintained_constants_betensor,
            so if kk is the saved weight.dtype, we skip it and set it to weight.dtype when kk == weights
            '''
            if kk.endswith('_dtype'):
                continue
            self.constants[kk].betensor = vv
            if f"{kk}_dtype" in maintained_constants_betensor:
                self.constants[kk].dtype = maintained_constants_betensor[f"{kk}_dtype"]
        # restore quantized flag and constants
        self.quantized = quant_state
        if None != constants_betensor_backup:
            for k, t in constants_betensor_backup.items():
                self.constants[k].betensor = t
        if None != params_backup:
            _ = _node_params_replace(self.params, params_backup, replace_keys)
        if True == self.quantized:
            if 'debug_fake_quantize' in self.attrs and self.attrs['debug_fake_quantize']:
                for out in self.outputs:
                    out.debug_flag = 1
            else:
                for out in self.outputs:
                    out.debug_flag = 0

        # fit outputs dtype
        if self.fit_dtype_enabled:
            for t in self.outputs:
                t.fit_dtype()

        # check shape consistency
        for t in self.outputs:
            ori_dshape = t.betensor.shape
            ori_sshape = ori_dshape
            if t.ir_shape is not None:
                ori_sshape = t.ir_shape
            else:
                OPT_WARN(
                    f"IR Shape Info of Tensor: {t.name} in layer_id={self.attrs.get('layer_id', -1)} {self.type} is lost", log_once=True)

            dshape = list(ori_dshape)
            sshape = list(ori_sshape)
            if sshape != dshape:
                if len(sshape) != len(dshape):
                    dynamic_shape = [f"{s}?" for s in sshape]
                else:
                    dynamic_shape = [f"{ori_sshape[i]}?" if ori_sshape[i] != ori_dshape[i]
                                     else f"{ori_sshape[i]}" for i in range(len(ori_sshape))]
                    # in order to avoid batch_size dim, and defaultly the batch_size_dim == 0 or 1
                    if len(sshape) > 1:
                        bs_idx = 0 if self.get_attrs("batch_size_in_IR", optional=True, default_value=1) > 0 else 1
                        dshape.pop(bs_idx)
                        sshape.pop(bs_idx)
                if dshape != sshape:
                    OPT_WARN(f'Get inconformity tensor shape {ori_dshape} with the original IR {ori_sshape}, '
                             f'layer_id={self.attrs.get("layer_id", "-1")}, tensor_name={t.name} of in {self}', log_once=True)
                t.attrs['dynamic_shape'] = ori_dshape

        if self.forward_hook is not None:
            self.forward_hook(self)

        if self.graph:
            tz = PyTensor('null').betensor
            # reduce tensor's reference count
            for it in self.inputs:
                if it.name in self.graph.ref_count_tensors.keys():
                    self.graph.ref_count_tensors[it.name][0] -= 1
            # clear useless tensors out of cache for memory saving
            count_free = 0
            for rkey, rval in self.graph.ref_count_tensors.items():
                rt = rval[1]
                if 0 == rval[0] and rt not in self.graph.output_tensors:
                    del rt.betensor
                    rt.betensor = tz
                    count_free += 1

        return ret

    def quantize(self, *args, **kwargs):
        ret = None
        from AIPUBuilder.Optimizer.framework import QUANT_OP_DICT, OpType
        from AIPUBuilder.Optimizer.utils.dtype_utils import is_signed, dtype2str
        from AIPUBuilder.Optimizer.logger import OPT_FATAL, OPT_DEBUG, OPT_ERROR, OPT_WARN

        import torch

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
        # self.attrs['params_original'] = self.params_clone()
        self.attrs['params_original'] = self.params.clone()

        for k, t in self.constants.items():
            key_tensors.append(t)
        if 'map_to_original_node' in self.attrs.keys():
            self.attrs['constants_betensor_original'] = {}
            for k, t in self.constants.items():
                tc = None
                qnmap = self.attrs['map_to_original_node']
                if self.name in qnmap.keys():
                    # reduce memory
                    tc = qnmap[self.name].constants[k].betensor if k in qnmap[self.name].constants.keys() else None
                else:
                    OPT_DEBUG(
                        f"layer_id={self.attrs['layer_id']}, type={self.type}, name={self.name} can not find corresponding original node", log_once=True)
                    tc = t.betensor.clone().detach()
                self.attrs['constants_betensor_original'][k] = tc
        else:
            # 'constants_betensor_original' will not in self.attrs, so an error can raise up when it's needed by not correctly evaluated
            pass
        for t in self.placeholders:
            key_tensors.append(t)

        if not unquantifiable:
            for t in key_tensors:
                need_not_clear_scale = (self.type == OpType.FullyConnected) and all(
                    sname in self.constants.keys() for sname in ['scale0', 'scale1'])
                if (not need_not_clear_scale):
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
        # replace quantized source weights/biases if adaquant optimization was applied
        if 'weights' in self.constants and 'adaquant_weights' in self.attrs:
            bkey = self.attrs['q_bits_weight']
            qnw = self.constants['weights']
            if bkey in self.attrs['adaquant_weights'].keys():
                OPT_DEBUG('Adaquant optimization was applied on layer_id=%s, %s, %s' %
                          (str(self.attrs['layer_id']), str(self.type), self.name))
                qnw.betensor = self.attrs['adaquant_weights'][bkey]
        if 'biases' in self.constants and 'adaquant_biases' in self.attrs:
            bkey = self.attrs['q_bits_bias']
            qnw = self.constants['biases']
            if bkey in self.attrs['adaquant_biases'].keys():
                OPT_DEBUG('Adaquant optimization was applied on layer_id=%s, %s, %s' %
                          (str(self.attrs['layer_id']), str(self.type), self.name))
                qnw.betensor = self.attrs['adaquant_biases'][bkey]
        ####################
        # replace quantized weights if gptq optimization was applied
        if 'weights' in self.constants and 'gptq_weights' in self.attrs:
            bkey = self.attrs['q_bits_weight']
            qnw = self.constants['weights']
            if bkey in self.attrs['gptq_weights'].keys():
                OPT_DEBUG('gptq optimization was applied on layer_id=%s, %s, %s' %
                          (str(self.attrs['layer_id']), str(self.type), self.name))
                qnw.betensor = self.attrs['gptq_weights'][bkey]
        ####################
        # then do the quantization
        ret = QUANT_OP_DICT[self.type](self, *args, **kwargs)
        # finnaly check properties that must be decided during quantization
        for t in (list(self.constants.values()) + list(self.outputs)):
            if t not in key_tensors or 'scale0' in t.name or 'scale1' in t.name or 'zp0' in t.name:
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

    def approximate(self, *args, **kwargs):
        from AIPUBuilder.Optimizer.framework import APPROX_OP_DICT, OpType
        from AIPUBuilder.Optimizer.logger import OPT_DEBUG
        # in the future, may add a IR field 'is_perf_mode=True' when an OP does not has an approximate interface,
        # which will force to choose an original high precision lib implementation
        name_func_dict = {k.name.lower(): v for k, v in APPROX_OP_DICT.items()}
        func = None
        if self.type in APPROX_OP_DICT:
            func = APPROX_OP_DICT[self.type]
        elif OpType.Activation == self.type:
            mkey = self.params['method'].lower()
            func = name_func_dict[mkey] if mkey in name_func_dict else None
        if func is None:
            OPT_DEBUG('op "%s" has not registered an approximate function' % str(self.type), log_once=True)
            return
        func(self, *args, **kwargs)

    @property
    def is_dynamic(self):
        return True if 'enable_ds' in self.attrs and self.attrs['enable_ds'] else False

    def get_lib_dtype_spec(self):
        from AIPUBuilder.Optimizer.framework.pycore.pytype import OpType
        if self.type in [OpType.Reshape, OpType.Concat]:
            return []
        elif self.type in PyNode.all_op_dtype_spec:
            return PyNode.all_op_dtype_spec[self.type]
        else:
            spec = get_op_dtype_spec(self.type)
            PyNode.all_op_dtype_spec[self.type] = spec
            return spec

    def param_show(self, *args):

        from AIPUBuilder.Optimizer.logger import OPT_DEBUG, OPT_ERROR
        import torch

        def show_output_min_max_scale(id):
            OPT_DEBUG(
                f'\t{id}. output min/max: [min, max, mean, std, scale, dtype]:')
            try:
                for out in self.outputs:
                    out_scale = out.scale[:10] if out.is_perchannel_scales() else out.scale
                    out_zerop = out.zerop[:10] if out.is_perchannel_scales() else out.zerop
                    OPT_DEBUG(f'\t\t{out.name}:')
                    OPT_DEBUG(
                        f'\t\t[{out.min}, {out.max}, {out.running_mean: <.6f}, {out.running_std: <.6f}, '
                        f'{out_scale}, {out_zerop}, {str(out.dtype)}]')
            except Exception as e:
                from AIPUBuilder.Optimizer.logger import OPT_ERROR
                OPT_ERROR(f"{self}, {out}")
                raise e

        def show_constant_min_max_scale(id):
            if self.constants:
                OPT_DEBUG(
                    f'\t{id}. constant min/max: [min, max, mean, std, qmin_v, qmax_v, dtype]:')
                for k, c in self.constants.items():
                    qmax_v = torch.max(c.betensor).item()
                    qmin_v = torch.min(c.betensor).item()
                    OPT_DEBUG(f'\t\t{k}:')
                    OPT_DEBUG(
                        f'\t\t[{c.min.item(): <.6f}, {c.max.item(): <.6f}, {c.running_mean}, {c.running_std}, {qmin_v}, {qmax_v}, {str(c.dtype)}]')

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
                    t_scale = tensor.scale[:10] if tensor.is_perchannel_scales() else tensor.scale
                    t_zerop = tensor.zerop[:10] if tensor.is_perchannel_zerops() else tensor.zerop
                    pstr += f'[{ele}] scale:{str(t_scale)}, zp: {str(t_zerop)} bits:{str(tensor.qbits)} '
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

    def __repr__(self):
        return (f"'node info: name={self.name}, type={self.type}, "
                f"layer_id = {self.get_attrs('layer_id', optional=True, default_value='unknow')}'")

    def set_ir_field(self, ir_field_name, ir_data, ir_data_type=None):
        import torch
        from AIPUBuilder.Optimizer.utils import (range2dtype, dtype2nptype, is_float, dtype2torch_type,
                                                 is_torch_tensor_with_multi_data, is_torch_tensor)
        from AIPUBuilder.Optimizer.framework import PyTensor

        def _dtype(dtype):
            return int if not is_float(dtype) else float

        if is_torch_tensor_with_multi_data(ir_data):
            if ir_data_type is None:
                ir_data_type = range2dtype(ir_data.min(), ir_data.max(), False if ir_data.min() >= 0 else True)[1]
            self.constants[ir_field_name] = PyTensor(f"{self.name}_{ir_field_name}", ir_data, ir_data_type)
        elif is_torch_tensor(ir_data):
            if ir_data_type is None:
                self.params[ir_field_name] = int(ir_data.item())
            else:
                self.params[ir_field_name] = ir_data.to(dtype2torch_type(ir_data_type)).item()
        elif isinstance(ir_data, (list, tuple)):
            self.params[ir_field_name] = ir_data
        elif isinstance(ir_data, (int, float)):
            self.params[ir_field_name] = _dtype(ir_data_type)(ir_data) if ir_data_type is not None else ir_data
        else:
            self.params[ir_field_name] = ir_data

    def get_ir_field(self, ir_field_names, default_value=None):
        from AIPUBuilder.Optimizer.logger import OPT_ERROR, OPT_WARN, OPT_DEBUG
        ret = None
        if not isinstance(ir_field_names, (tuple, list)) and isinstance(ir_field_names, str):
            ir_field_names = [ir_field_names, ]
        for ir_field_name in ir_field_names:
            if ir_field_name in self.constants:
                ret = self.constants[ir_field_name].betensor
            elif ir_field_name in self.params:
                ret = self.params[ir_field_name]
            if ret is not None:
                break
        if ret is None:
            OPT_DEBUG(f"ir_field={ir_field_name} neither in node.constants nor in node.params, "
                      f"and return default value={default_value}.")
            ret = default_value
        return ret

    @staticmethod
    def deduce_quantization_infos(node):
        from AIPUBuilder.Optimizer.utils import is_float, dtype2range, dtype2bits
        from AIPUBuilder.Optimizer.framework import OpType, get_tensor_default_property
        from AIPUBuilder.Optimizer.logger import OPT_ERROR, OPT_WARN
        import torch

        def _deduce_quantization_info_to_tensor_from_ir(node, updated_fields):
            in_out_tensors = [*node.inputs, *node.outputs]
            key_axes = node.get_param('activation_quantization_axis',
                                      optional=True, default_value=[None]*len(node.outputs))
            key_axes = [None if isinstance(ka, str) and ka.lower() == 'none' else ka for ka in key_axes]
            for t in in_out_tensors:
                if is_float(t.dtype) and dtype2bits(t.dtype) >= 16:
                    continue
                o_dtype = t.dtype
                if t.ir_range is None:
                    qmin, qmax = dtype2range(o_dtype)
                else:
                    qmin, qmax = t.ir_range[0], t.ir_range[1]
                qbits = dtype2bits(o_dtype)
                key_axis = key_axes[node.outputs.index(t)] if t in node.outputs else None
                quantization_infos = {
                    'qmin': qmin,
                    'qmax': qmax,
                    'qbits': qbits,
                }
                for field in updated_fields:
                    if field in quantization_infos.keys():
                        t.__setattr__(field, quantization_infos[field])
                    if t in node.outputs:
                        t.key_axis = key_axis

        if node is None:
            OPT_ERROR("please check the node(==None) before deduce quantization information.")
            return

        node.quantized = True
        if node.type in [OpType.Quantize]:
            _deduce_quantization_info_to_tensor_from_ir(node, get_tensor_default_property())
            return

        if node.get_param('unquantifiable', optional=True, default_value=False):
            node.quantized = False
            if node.get_param('is_perf_mode', optional=True, default_value=False):
                node.approximated = True
            if 'weights' in node.constants.keys():
                wt = node.constants["weights"]
                if not (is_float(wt.dtype) and dtype2bits(wt.dtype) >= 16):
                    node.attrs['weight_only_quantization'] = True
        else:
            with_fpx = False
            dtypes = [t.dtype for t in (list(node.outputs) + list(node.inputs))]
            for dt in dtypes:
                if is_float(dt) and dtype2bits(dt) <= 8:
                    with_fpx = True
                    break
            if not with_fpx:
                with_lut = False
                constants_name = node.constants.keys()
                for name in constants_name:
                    if 'lut' in name:
                        with_lut = True
                        if is_float(node.constants[name].dtype):
                            OPT_WARN(
                                f"{node},constant[{name}] dtype is float, currently set node.quantized = True.")
                        break
                    else:
                        dtypes.append(node.constants[name].dtype)
                if not with_lut:
                    for dt in dtypes:
                        if is_float(dt) and dtype2bits(dt) >= 16:
                            node.quantized = False
                            break

        if 'weight_block_size' not in node.params:
            for k, v in node.constants.items():
                if k not in ['weights', 'biases']:
                    continue
                scales = v.scale
                zps = v.zerop
                ir_shape_len = len(v.ir_shape)
                ka = v.key_axis
                if ir_shape_len and ir_shape_len - 1 >= v.key_axis:
                    channel_num = v.ir_shape[v.key_axis]
                    if scales.numel() == zps.numel() and scales.numel() > 1 and scales.numel() > channel_num:
                        ic = node.inputs[0].ir_shape[-1]
                        group_size = ic // (scales.numel() // channel_num)
                        node.params['weight_block_size'] = group_size
                        break

        if node.quantized:
            _deduce_quantization_info_to_tensor_from_ir(node, get_tensor_default_property())


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


def _is_approximated(self):
    return "approximated" in self.attrs and self.attrs["approximated"]


def _set_approximated(self, flag):
    self.attrs["approximated"] = flag


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


def _whether_enable_fit_dtype(self):
    return "enable_fit_dtype" in self.attrs and self.attrs["enable_fit_dtype"]


def _set_enable_fit_dtype(self, flag):
    self.attrs["enable_fit_dtype"] = flag


def _whether_unquantifiable(self):
    return 'unquantifiable' in self.params and self.params['unquantifiable']


def _set_unquantifiable(self, flag):
    self.params['unquantifiable'] = flag


PyNode.current_batch_size = property(_get_current_batch_size, _set_current_batch_size)
PyNode.current_batch_idx = property(_get_current_batch_idx, _set_current_batch_idx)
# wether the node will use quantized int forward implementation
PyNode.quantized = property(_is_quantized, _set_quantized)
# wether the node will use approximated float forward implementation
PyNode.approximated = property(_is_approximated, _set_approximated)
PyNode.additional = property(_is_additional, _set_additional)
PyNode.force_dtype_int = property(_whether_force_dtype_int, _set_force_dtype_int)
PyNode.force_shift_positive = property(_whether_force_shift_positive, _set_force_shift_positive)
PyNode.fit_dtype_enabled = property(_whether_enable_fit_dtype, _set_enable_fit_dtype)
PyNode.unquantifiable = property(_whether_unquantifiable, _set_unquantifiable)


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
        dt_dict = {str(dt).upper(): dt for dt in Dtype}
        dt_dict['DTYPE.FP8E4M3FN'] = Dtype.FP8_E4M3FN
        dt_dict['DTYPE.FP8E5M2'] = Dtype.FP8_E5M2

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
                spec.in_dtypes.append(dt_dict[s.name().upper()])
            for s in c_spec.out_dtypes:
                spec.out_dtypes.append(dt_dict[s.name().upper()])
            dtype_specs.append(spec)

        """
        when importing get_dtype of _C interface is successful, but calling get_dtype returns a ERROR log, which leading
        to len(dtype_specs) == 0.
        """
        if len(dtype_specs) == 0:
            OPT_WARN(f"Using _C get_dtype donot find the {ntype}'s dtype, so try to use lookup static table "
                     f"which may not be up to date.", log_once=True)
            return lookup_static_op_dtype_spec_table(ntype)
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
def lookup_static_op_dtype_spec_table(ntype):
    static_all_op_dtype_spec = {'OpType.Abs': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['UINT16']), (['INT8'], ['UINT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.AccidentalHits': [(['BFLOAT16', 'BFLOAT16'], ['UINT16', 'UINT16', 'UINT16']), (['FLOAT16', 'FLOAT16'], ['UINT16', 'UINT16', 'UINT16']), (['INT16', 'INT16'], ['UINT16', 'UINT16', 'UINT16']), (['INT32', 'INT32'], ['UINT16', 'UINT16', 'UINT16']), (['INT8', 'INT8'], ['UINT16', 'UINT16', 'UINT16']), (['UINT16', 'UINT16'], ['UINT16', 'UINT16', 'UINT16']), (['UINT32', 'UINT32'], ['UINT16', 'UINT16', 'UINT16']), (['UINT8', 'UINT8'], ['UINT16', 'UINT16', 'UINT16'])], 'OpType.Acos': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Acosh': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Activation': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Add': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT16']), (['INT16', 'INT16'], ['UINT8']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT16'], ['INT8']), (['INT16', 'UINT16'], ['UINT16']), (['INT16', 'UINT16'], ['UINT8']), (['INT32', 'INT32'], ['INT32']), (['INT8', 'INT8'], ['INT16']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT16']), (['INT8', 'INT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT16']), (['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT8'], ['UINT16']), (['INT8', 'UINT8'], ['UINT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT16'], ['UINT8']), (['UINT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['INT8']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT16'], ['UINT8']), (['UINT32', 'UINT32'], ['UINT32']), (['UINT8', 'INT8'], ['INT16']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT16']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT16']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT16']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.ArgMinMax': [(['BFLOAT16'], ['INT32']), (['BFLOAT16'], ['INT8']), (['BFLOAT16'], ['UINT16']), (['BFLOAT16'], ['UINT32']), (['BFLOAT16'], ['UINT8']), (['FLOAT16'], ['INT32']), (['FLOAT16'], ['INT8']), (['FLOAT16'], ['UINT16']), (['FLOAT16'], ['UINT32']), (['FLOAT16'], ['UINT8']), (['INT16'], ['INT16']), (['INT16'], ['INT32']), (['INT16'], ['INT8']), (['INT16'], ['UINT16']), (['INT16'], ['UINT32']), (['INT16'], ['UINT8']), (['INT32'], ['INT32']), (['INT32'], ['UINT16']), (['INT32'], ['UINT32']), (['INT8'], ['INT16']), (['INT8'], ['INT32']), (['INT8'], ['INT8']), (['INT8'], ['UINT16']), (['INT8'], ['UINT32']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['INT32']), (['UINT16'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT16'], ['UINT32']), (['UINT16'], ['UINT8']), (['UINT32'], ['INT32']), (['UINT32'], ['UINT16']), (['UINT32'], ['UINT32']), (['UINT8'], ['INT16']), (['UINT8'], ['INT32']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT16']), (['UINT8'], ['UINT32']), (['UINT8'], ['UINT8'])], 'OpType.Asin': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Asinh': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Atan': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.BNLL': [(['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.BasicLSTM': [(['BFLOAT16', 'BFLOAT16', 'BFLOAT16'], ['BFLOAT16', 'BFLOAT16', 'BFLOAT16']), (['BFLOAT16', 'BFLOAT16', 'BFLOAT16'], ['BFLOAT16', 'BFLOAT16']), (['BFLOAT16', 'BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'FLOAT16', 'FLOAT16'], ['FLOAT16', 'FLOAT16', 'FLOAT16']), (['FLOAT16', 'FLOAT16', 'FLOAT16'], ['FLOAT16', 'FLOAT16']), (['FLOAT16', 'FLOAT16', 'FLOAT16'], ['FLOAT16']), (['INT16', 'INT16', 'INT16'], ['INT16', 'INT16', 'INT16']), (['INT16', 'INT16', 'INT16'], ['INT16', 'INT16']), (['INT16', 'INT16', 'INT16'], ['INT16']), (['INT8', 'INT8', 'INT8'], ['INT8', 'INT8', 'INT8']), (['INT8', 'INT8', 'INT8'], ['INT8', 'INT8']), (['INT8', 'INT8', 'INT8'], ['INT8']), (['UINT16', 'INT16', 'INT16'], ['INT16', 'INT16', 'INT16']), (['UINT16', 'INT16', 'INT16'], ['INT16', 'INT16']), (['UINT16', 'INT16', 'INT16'], ['INT16']), (['UINT8', 'INT8', 'INT8'], ['INT8', 'INT8', 'INT8']), (['UINT8', 'INT8', 'INT8'], ['INT8', 'INT8']), (['UINT8', 'INT8', 'INT8'], ['INT8'])], 'OpType.BatchNorm': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.BatchToDepth': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.BatchToSpace': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.BiasAdd': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.BitShift': [(['INT16', 'UINT8'], ['INT16']), (['INT32', 'UINT8'], ['INT32']), (['INT8', 'UINT8'], ['INT8']), (['UINT16', 'UINT8'], ['UINT16']), (['UINT32', 'UINT8'], ['UINT32']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.BoundingBox': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16']), (['INT16', 'INT8'], ['INT16'])], 'OpType.CELU': [], 'OpType.CRELU': [], 'OpType.CTCBeamDecoder': [], 'OpType.CTCGreedyDecoder': [(['INT8', 'UINT16'], ['UINT16']), (['INT8', 'UINT8'], ['UINT16']), (['UINT8', 'UINT16'], ['UINT16']), (['UINT8', 'UINT8'], ['UINT16'])], 'OpType.CacheUpdate': [], 'OpType.Cast': [(['BFLOAT16'], ['FLOAT32']), (['BFLOAT16'], ['INT16']), (['BFLOAT16'], ['INT32']), (['BFLOAT16'], ['INT8']), (['BFLOAT16'], ['UINT16']), (['BFLOAT16'], ['UINT8']), (['FLOAT16'], ['FLOAT32']), (['FLOAT16'], ['INT16']), (['FLOAT16'], ['INT32']), (['FLOAT16'], ['INT8']), (['FLOAT16'], ['UINT16']), (['FLOAT16'], ['UINT8']), (['FLOAT32'], ['BFLOAT16']), (['FLOAT32'], ['FLOAT16']), (['INT16'], ['BFLOAT16']), (['INT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['INT32']), (['INT16'], ['INT8']), (['INT16'], ['UINT16']), (['INT16'], ['UINT32']), (['INT16'], ['UINT8']), (['INT32'], ['BFLOAT16']), (['INT32'], ['FLOAT16']), (['INT32'], ['INT16']), (['INT32'], ['INT32']), (['INT32'], ['INT8']), (['INT32'], ['UINT16']), (['INT32'], ['UINT32']), (['INT32'], ['UINT8']), (['INT8'], ['BFLOAT16']), (['INT8'], ['FLOAT16']), (['INT8'], ['INT16']), (['INT8'], ['INT32']), (['INT8'], ['INT8']), (['INT8'], ['UINT16']), (['INT8'], ['UINT32']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['INT32']), (['UINT16'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT16'], ['UINT32']), (['UINT16'], ['UINT8']), (['UINT8'], ['BFLOAT16']), (['UINT8'], ['FLOAT16']), (['UINT8'], ['INT16']), (['UINT8'], ['INT32']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT16']), (['UINT8'], ['UINT32']), (['UINT8'], ['UINT8'])], 'OpType.Ceil': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.ChannelShuffle': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.Clip': [], 'OpType.Compress': [(['BFLOAT16', 'INT16'], ['BFLOAT16']), (['BFLOAT16', 'INT8'], ['BFLOAT16']), (['BFLOAT16', 'UINT16'], ['BFLOAT16']), (['BFLOAT16', 'UINT8'], ['BFLOAT16']), (['FLOAT16', 'INT16'], ['FLOAT16']), (['FLOAT16', 'INT8'], ['FLOAT16']), (['FLOAT16', 'UINT16'], ['FLOAT16']), (['FLOAT16', 'UINT8'], ['FLOAT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT8'], ['INT16']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT8'], ['INT16']), (['INT8', 'INT16'], ['INT8']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'UINT16'], ['INT8']), (['INT8', 'UINT8'], ['INT8']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT8'], ['UINT16']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT8'], ['UINT16']), (['UINT8', 'INT16'], ['UINT8']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT16'], ['UINT8']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.Concat': [], 'OpType.Constant': [([], ['BFLOAT16']), ([], ['FLOAT16']), ([], ['INT16']), ([], ['INT32']), ([], ['INT8']), ([], ['UINT16']), ([], ['UINT32']), ([], ['UINT8'])], 'OpType.ConvInteger': [], 'OpType.ConvTranspose': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.ConvTranspose3D': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Convolution': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['FLOAT8_E4M3FN'], ['FLOAT8_E4M3FN']), (['FLOAT8_E4M3FN'], ['FLOAT8_E5M2']), (['FLOAT8_E5M2'], ['FLOAT8_E4M3FN']), (['FLOAT8_E5M2'], ['FLOAT8_E5M2']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Convolution3D': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Cosh': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Cosine': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Count': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT16'])], 'OpType.Crop': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT32'], ['INT32']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT32'], ['UINT32']), (['UINT8'], ['UINT8'])], 'OpType.CropAndResize': [(['INT16', 'UINT16', 'UINT8'], ['INT16']), (['INT8', 'UINT16', 'UINT8'], ['INT8']), (['INT8', 'UINT16', 'UINT8'], ['UINT8']), (['UINT8', 'UINT16', 'UINT8'], ['UINT8'])], 'OpType.DataStride': [], 'OpType.DeQuantize': [(['INT16'], ['BFLOAT16']), (['INT16'], ['FLOAT16']), (['INT16'], ['FLOAT32']), (['INT32'], ['FLOAT32']), (['INT8'], ['BFLOAT16']), (['INT8'], ['FLOAT16']), (['INT8'], ['FLOAT32']), (['UINT16'], ['BFLOAT16']), (['UINT16'], ['FLOAT16']), (['UINT16'], ['FLOAT32']), (['UINT8'], ['BFLOAT16']), (['UINT8'], ['FLOAT16']), (['UINT8'], ['FLOAT32'])], 'OpType.DecodeBox': [(['UINT8', 'INT8'], ['INT16', 'UINT16', 'UINT16', 'UINT8', 'UINT16'])], 'OpType.DepthToSpace': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.DepthwiseConv': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.DetectionOutput': [(['UINT8', 'INT8', 'INT16'], ['UINT8', 'INT16', 'UINT16', 'UINT16', 'UINT16'])], 'OpType.Div': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'UINT8'], ['INT8']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.ELU': [], 'OpType.Eltwise': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT16']), (['INT16', 'INT16'], ['UINT8']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT16'], ['INT8']), (['INT16', 'UINT16'], ['UINT16']), (['INT16', 'UINT16'], ['UINT8']), (['INT32', 'INT32'], ['INT32']), (['INT8', 'INT8'], ['INT16']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT16']), (['INT8', 'INT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT16']), (['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT8'], ['UINT16']), (['INT8', 'UINT8'], ['UINT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT16'], ['UINT8']), (['UINT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['INT8']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT16'], ['UINT8']), (['UINT8', 'INT8'], ['INT16']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT16']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT16']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT16']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.Erf': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Exp': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.FakeQuantWithMinMaxVars': [], 'OpType.Filter': [], 'OpType.Floor': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.FractionalPool': [(['FLOAT16'], ['FLOAT16', 'INT16', 'INT16']), (['INT16'], ['INT16', 'INT16', 'INT16']), (['INT8'], ['INT8', 'INT16', 'INT16']), (['UINT16'], ['UINT16', 'INT16', 'INT16']), (['UINT8'], ['UINT8', 'INT16', 'INT16'])], 'OpType.FullyConnected': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['FLOAT8_E4M3FN'], ['FLOAT8_E4M3FN']), (['FLOAT8_E4M3FN'], ['FLOAT8_E5M2']), (['FLOAT8_E5M2'], ['FLOAT8_E4M3FN']), (['FLOAT8_E5M2'], ['FLOAT8_E5M2']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.GELU': [], 'OpType.GRUv1': [(['INT16', 'INT16'], ['INT16']), (['INT8', 'INT8'], ['INT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT8', 'INT8'], ['INT8'])], 'OpType.GRUv3': [(['INT16', 'INT16'], ['INT16']), (['INT8', 'INT8'], ['INT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT8', 'INT8'], ['INT8'])], 'OpType.Gather': [(['BFLOAT16', 'INT16'], ['BFLOAT16']), (['BFLOAT16', 'INT32'], ['BFLOAT16']), (['BFLOAT16', 'INT8'], ['BFLOAT16']), (['BFLOAT16', 'UINT16'], ['BFLOAT16']), (['BFLOAT16', 'UINT32'], ['BFLOAT16']), (['BFLOAT16', 'UINT8'], ['BFLOAT16']), (['FLOAT16', 'INT16'], ['FLOAT16']), (['FLOAT16', 'INT32'], ['FLOAT16']), (['FLOAT16', 'INT8'], ['FLOAT16']), (['FLOAT16', 'UINT16'], ['FLOAT16']), (['FLOAT16', 'UINT32'], ['FLOAT16']), (['FLOAT16', 'UINT8'], ['FLOAT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT32'], ['INT16']), (['INT16', 'INT8'], ['INT16']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT32'], ['INT16']), (['INT16', 'UINT8'], ['INT16']), (['INT32', 'INT16'], ['INT32']), (['INT32', 'INT32'], ['INT32']), (['INT32', 'INT8'], ['INT32']), (['INT32', 'UINT16'], ['INT32']), (['INT32', 'UINT32'], ['INT32']), (['INT32', 'UINT8'], ['INT32']), (['INT8', 'INT16'], ['INT8']), (['INT8', 'INT32'], ['INT8']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'UINT16'], ['INT8']), (['INT8', 'UINT32'], ['INT8']), (['INT8', 'UINT8'], ['INT8']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT32'], ['UINT16']), (['UINT16', 'INT8'], ['UINT16']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT32'], ['UINT16']), (['UINT16', 'UINT8'], ['UINT16']), (['UINT32', 'INT16'], ['UINT32']), (['UINT32', 'INT32'], ['UINT32']), (['UINT32', 'INT8'], ['UINT32']), (['UINT32', 'UINT16'], ['UINT32']), (['UINT32', 'UINT32'], ['UINT32']), (['UINT32', 'UINT8'], ['UINT32']), (['UINT8', 'INT16'], ['UINT8']), (['UINT8', 'INT32'], ['UINT8']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT16'], ['UINT8']), (['UINT8', 'UINT32'], ['UINT8']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.GatherElements': [(['BFLOAT16', 'INT16'], ['BFLOAT16']), (['BFLOAT16', 'INT32'], ['BFLOAT16']), (['BFLOAT16', 'INT8'], ['BFLOAT16']), (['BFLOAT16', 'UINT16'], ['BFLOAT16']), (['BFLOAT16', 'UINT32'], ['BFLOAT16']), (['BFLOAT16', 'UINT8'], ['BFLOAT16']), (['FLOAT16', 'INT16'], ['FLOAT16']), (['FLOAT16', 'INT32'], ['FLOAT16']), (['FLOAT16', 'INT8'], ['FLOAT16']), (['FLOAT16', 'UINT16'], ['FLOAT16']), (['FLOAT16', 'UINT32'], ['FLOAT16']), (['FLOAT16', 'UINT8'], ['FLOAT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT32'], ['INT16']), (['INT16', 'INT8'], ['INT16']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT32'], ['INT16']), (['INT16', 'UINT8'], ['INT16']), (['INT8', 'INT16'], ['INT8']), (['INT8', 'INT32'], ['INT8']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'UINT16'], ['INT8']), (['INT8', 'UINT32'], ['INT8']), (['INT8', 'UINT8'], ['INT8']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT32'], ['UINT16']), (['UINT16', 'INT8'], ['UINT16']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT32'], ['UINT16']), (['UINT16', 'UINT8'], ['UINT16']), (['UINT8', 'INT16'], ['UINT8']), (['UINT8', 'INT32'], ['UINT8']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT16'], ['UINT8']), (['UINT8', 'UINT32'], ['UINT8']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.GatherND': [(['BFLOAT16', 'INT16'], ['BFLOAT16']), (['BFLOAT16', 'INT32'], ['BFLOAT16']), (['BFLOAT16', 'INT8'], ['BFLOAT16']), (['BFLOAT16', 'UINT16'], ['BFLOAT16']), (['BFLOAT16', 'UINT32'], ['BFLOAT16']), (['BFLOAT16', 'UINT8'], ['BFLOAT16']), (['FLOAT16', 'INT16'], ['FLOAT16']), (['FLOAT16', 'INT32'], ['FLOAT16']), (['FLOAT16', 'INT8'], ['FLOAT16']), (['FLOAT16', 'UINT16'], ['FLOAT16']), (['FLOAT16', 'UINT32'], ['FLOAT16']), (['FLOAT16', 'UINT8'], ['FLOAT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT32'], ['INT16']), (['INT16', 'INT8'], ['INT16']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT32'], ['INT16']), (['INT16', 'UINT8'], ['INT16']), (['INT8', 'INT16'], ['INT8']), (['INT8', 'INT32'], ['INT8']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'UINT16'], ['INT8']), (['INT8', 'UINT32'], ['INT8']), (['INT8', 'UINT8'], ['INT8']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT32'], ['UINT16']), (['UINT16', 'INT8'], ['UINT16']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT32'], ['UINT16']), (['UINT16', 'UINT8'], ['UINT16']), (['UINT8', 'INT16'], ['UINT8']), (['UINT8', 'INT32'], ['UINT8']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT16'], ['UINT8']), (['UINT8', 'UINT32'], ['UINT8']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.Gemm': [(['BFLOAT16', 'BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'FLOAT16', 'FLOAT16'], ['FLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16'])], 'OpType.GenerateProposals': [(['BFLOAT16', 'BFLOAT16', 'BFLOAT16'], ['BFLOAT16', 'BFLOAT16', 'INT32', 'INT32']), (['FLOAT16', 'FLOAT16', 'FLOAT16'], ['FLOAT16', 'FLOAT16', 'INT32', 'INT32']), (['UINT8', 'UINT8', 'UINT8'], ['UINT8', 'UINT16', 'UINT16', 'UINT16'])], 'OpType.GlobalPool': [], 'OpType.GridSample': [(['INT8', 'INT16'], ['INT8'])], 'OpType.GroupNorm': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.HardSigmoid': [], 'OpType.Hardmax': [], 'OpType.Hardswish': [], 'OpType.InTopK': [(['BFLOAT16', 'UINT16'], ['UINT8']), (['BFLOAT16', 'UINT32'], ['UINT8']), (['BFLOAT16', 'UINT8'], ['UINT8']), (['FLOAT16', 'UINT16'], ['UINT8']), (['FLOAT16', 'UINT32'], ['UINT8']), (['FLOAT16', 'UINT8'], ['UINT8']), (['INT16', 'UINT16'], ['UINT8']), (['INT16', 'UINT32'], ['UINT8']), (['INT16', 'UINT8'], ['UINT8']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT8']), (['UINT16', 'UINT16'], ['UINT8']), (['UINT16', 'UINT32'], ['UINT8']), (['UINT16', 'UINT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.Input': [], 'OpType.InstanceNorm': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Interp': [], 'OpType.LRN': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.LayerNorm': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.LeakyRELU': [], 'OpType.Log': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.LogSoftmax': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['INT16']), (['UINT8'], ['INT8'])], 'OpType.Logical': [(['BFLOAT16', 'BFLOAT16'], ['INT16']), (['BFLOAT16', 'BFLOAT16'], ['INT8']), (['BFLOAT16', 'BFLOAT16'], ['UINT16']), (['BFLOAT16', 'BFLOAT16'], ['UINT8']), (['FLOAT16', 'FLOAT16'], ['INT16']), (['FLOAT16', 'FLOAT16'], ['INT8']), (['FLOAT16', 'FLOAT16'], ['UINT16']), (['FLOAT16', 'FLOAT16'], ['UINT8']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT16']), (['INT16', 'INT16'], ['UINT8']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT16'], [
        'INT8']), (['INT16', 'UINT16'], ['UINT16']), (['INT16', 'UINT16'], ['UINT8']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT8'], ['UINT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT16'], ['UINT8']), (['UINT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['INT8']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT16'], ['UINT8']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.MISH': [], 'OpType.MVN': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.MatMul': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['BFLOAT16', 'BFLOAT16'], ['FLOAT16']), (['BFLOAT16', 'FLOAT16'], ['BFLOAT16']), (['BFLOAT16', 'FLOAT16'], ['FLOAT16']), (['FLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'BFLOAT16'], ['FLOAT16']), (['FLOAT16', 'FLOAT16'], ['BFLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16']), (['FLOAT8_E4M3FN', 'FLOAT8_E4M3FN'], ['FLOAT8_E4M3FN']), (['FLOAT8_E4M3FN', 'FLOAT8_E4M3FN'], ['FLOAT8_E5M2']), (['FLOAT8_E4M3FN', 'FLOAT8_E5M2'], ['FLOAT8_E4M3FN']), (['FLOAT8_E4M3FN', 'FLOAT8_E5M2'], ['FLOAT8_E5M2']), (['FLOAT8_E5M2', 'FLOAT8_E4M3FN'], ['FLOAT8_E4M3FN']), (['FLOAT8_E5M2', 'FLOAT8_E4M3FN'], ['FLOAT8_E5M2']), (['FLOAT8_E5M2', 'FLOAT8_E5M2'], ['FLOAT8_E4M3FN']), (['FLOAT8_E5M2', 'FLOAT8_E5M2'], ['FLOAT8_E5M2']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT16']), (['INT16', 'INT16'], ['UINT8']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT16'], ['INT8']), (['INT16', 'UINT16'], ['UINT16']), (['INT16', 'UINT16'], ['UINT8']), (['INT8', 'INT8'], ['INT16']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT16']), (['INT8', 'INT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT16']), (['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT8'], ['UINT16']), (['INT8', 'UINT8'], ['UINT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT16'], ['UINT8']), (['UINT8', 'INT8'], ['INT16']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT16']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT16']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT16']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.MatMulInteger': [], 'OpType.MaxPoolingWithArgMax': [(['BFLOAT16'], ['BFLOAT16', 'INT32']), (['FLOAT16'], ['FLOAT16', 'INT32']), (['INT16'], ['INT16', 'INT32']), (['INT8'], ['INT8', 'INT32']), (['UINT16'], ['UINT16', 'INT32']), (['UINT8'], ['UINT8', 'INT32'])], 'OpType.MaxRoiPool': [(['BFLOAT16', 'UINT16'], ['BFLOAT16']), (['BFLOAT16', 'UINT8'], ['BFLOAT16']), (['FLOAT16', 'UINT16'], ['FLOAT16']), (['FLOAT16', 'UINT8'], ['FLOAT16']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT8'], ['INT16']), (['INT8', 'UINT16'], ['INT8']), (['INT8', 'UINT8'], ['INT8']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT8'], ['UINT16']), (['UINT8', 'UINT16'], ['UINT8']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.MaxUnpool': [(['BFLOAT16', 'INT32'], ['BFLOAT16']), (['FLOAT16', 'INT32'], ['FLOAT16']), (['INT16', 'INT32'], ['INT16']), (['INT8', 'INT32'], ['INT8']), (['UINT16', 'INT32'], ['UINT16']), (['UINT8', 'INT32'], ['UINT8'])], 'OpType.Meshgrid': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16', 'BFLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16', 'FLOAT16']), (['INT16', 'INT16'], ['INT16', 'INT16']), (['INT16', 'UINT16'], ['INT16', 'UINT16']), (['INT8', 'INT8'], ['INT8', 'INT8']), (['INT8', 'UINT8'], ['INT8', 'UINT8']), (['UINT16', 'INT16'], ['UINT16', 'INT16']), (['UINT16', 'UINT16'], ['UINT16', 'UINT16']), (['UINT8', 'INT8'], ['UINT8', 'INT8']), (['UINT8', 'UINT8'], ['UINT8', 'UINT8'])], 'OpType.Mod': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16']), (['INT16', 'INT16'], ['INT16']), (['INT8', 'INT8'], ['INT8']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.Moments': [(['BFLOAT16'], ['BFLOAT16', 'BFLOAT16']), (['FLOAT16'], ['FLOAT16', 'FLOAT16']), (['INT16'], ['INT16', 'UINT16']), (['INT8'], ['INT8', 'UINT8']), (['UINT8'], ['UINT8', 'UINT8'])], 'OpType.Mul': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT16']), (['INT16', 'INT16'], ['UINT8']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT16'], ['INT8']), (['INT16', 'UINT16'], ['UINT16']), (['INT16', 'UINT16'], ['UINT8']), (['INT32', 'INT32'], ['INT32']), (['INT8', 'INT8'], ['INT16']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT16']), (['INT8', 'INT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT16']), (['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT8'], ['UINT16']), (['INT8', 'UINT8'], ['UINT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT16'], ['UINT8']), (['UINT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['INT8']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT16'], ['UINT8']), (['UINT32', 'UINT32'], ['UINT32']), (['UINT8', 'INT8'], ['INT16']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT16']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT16']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT16']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.NMS': [(['BFLOAT16', 'INT32', 'INT32', 'BFLOAT16'], ['BFLOAT16', 'INT32', 'BFLOAT16', 'INT32']), (['FLOAT16', 'INT32', 'INT32', 'FLOAT16'], ['FLOAT16', 'INT32', 'FLOAT16', 'INT32']), (['INT16', 'UINT16', 'INT16', 'UINT16'], ['INT16', 'UINT16', 'UINT16', 'UINT16']), (['INT16', 'UINT16', 'INT16', 'UINT8'], ['INT16', 'UINT16', 'UINT8', 'UINT16']), (['INT16', 'UINT16', 'UINT16', 'UINT16'], ['INT16', 'UINT16', 'UINT16', 'UINT16']), (['INT16', 'UINT16', 'UINT16', 'UINT8'], ['INT16', 'UINT16', 'UINT8', 'UINT16'])], 'OpType.Negative': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Normalization': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.OneHot': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['INT32']), (['INT16'], ['INT8']), (['INT16'], ['UINT16']), (['INT16'], ['UINT32']), (['INT16'], ['UINT8']), (['INT32'], ['INT16']), (['INT32'], ['INT32']), (['INT32'], ['INT8']), (['INT32'], ['UINT16']), (['INT32'], ['UINT32']), (['INT32'], ['UINT8']), (['INT8'], ['INT16']), (['INT8'], ['INT32']), (['INT8'], ['INT8']), (['INT8'], ['UINT16']), (['INT8'], ['UINT32']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['INT32']), (['UINT16'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT16'], ['UINT32']), (['UINT16'], ['UINT8']), (['UINT32'], ['INT16']), (['UINT32'], ['INT32']), (['UINT32'], ['INT8']), (['UINT32'], ['UINT16']), (['UINT32'], ['UINT32']), (['UINT32'], ['UINT8']), (['UINT8'], ['INT16']), (['UINT8'], ['INT32']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT16']), (['UINT8'], ['UINT32']), (['UINT8'], ['UINT8'])], 'OpType.OverlapAdd': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.PRELU': [], 'OpType.Pad': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.Permute': [], 'OpType.Pooling': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.Pooling3D': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.PostNMS1': [(['BFLOAT16', 'UINT16', 'UINT16'], ['BFLOAT16', 'BFLOAT16']), (['FLOAT16', 'UINT16', 'UINT16'], ['FLOAT16', 'FLOAT16']), (['INT16', 'UINT16', 'UINT16'], ['INT16', 'UINT16'])], 'OpType.PostNMS2': [], 'OpType.Pow': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'UINT16'], ['INT16']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'UINT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.Proposal': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16', 'BFLOAT16', 'INT32', 'INT32']), (['FLOAT16', 'FLOAT16'], ['FLOAT16', 'FLOAT16', 'INT32', 'INT32']), (['UINT8', 'INT8'], ['UINT8', 'INT16', 'UINT16', 'UINT16'])], 'OpType.PyramidROIAlign': [(['INT16', 'INT8', 'INT8', 'INT8', 'INT8'], ['INT8']), (['INT16', 'UINT8', 'UINT8', 'UINT8', 'UINT8'], ['UINT8']), (['UINT16', 'INT8', 'INT8', 'INT8', 'INT8'], ['INT8']), (['UINT16', 'UINT8', 'UINT8', 'UINT8', 'UINT8'], ['UINT8'])], 'OpType.Quantize': [(['BFLOAT16'], ['INT16']), (['BFLOAT16'], ['INT8']), (['BFLOAT16'], ['UINT16']), (['BFLOAT16'], ['UINT8']), (['FLOAT16'], ['INT16']), (['FLOAT16'], ['INT8']), (['FLOAT16'], ['UINT16']), (['FLOAT16'], ['UINT8']), (['FLOAT32'], ['INT16']), (['FLOAT32'], ['INT32']), (['FLOAT32'], ['INT8']), (['FLOAT32'], ['UINT16']), (['FLOAT32'], ['UINT8'])], 'OpType.RELU': [], 'OpType.RELU6': [], 'OpType.ROIPooling': [], 'OpType.Reciprocal': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Reduce': [(['BFLOAT16'], ['BFLOAT16']), (['BFLOAT16'], ['INT8']), (['FLOAT16'], ['FLOAT16']), (['FLOAT16'], ['INT8']), (['INT16'], ['INT16']), (['INT16'], ['INT8']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Region': [(['INT8'], ['UINT8', 'INT16', 'INT16', 'INT16', 'INT16'])], 'OpType.RegionFuse': [], 'OpType.Repeat': [], 'OpType.Reshape': [], 'OpType.Resize': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.ReverseSequence': [(['BFLOAT16', 'INT32'], ['BFLOAT16']), (['BFLOAT16', 'UINT16'], ['BFLOAT16']), (['BFLOAT16', 'UINT32'], ['BFLOAT16']), (['FLOAT16', 'INT32'], ['FLOAT16']), (['FLOAT16', 'UINT16'], ['FLOAT16']), (['FLOAT16', 'UINT32'], ['FLOAT16']), (['INT16', 'INT32'], ['INT16']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT32'], ['INT16']), (['INT8', 'INT32'], ['INT8']), (['INT8', 'UINT16'], ['INT8']), (['INT8', 'UINT32'], ['INT8']), (['UINT16', 'INT32'], ['UINT16']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT32'], ['UINT16']), (['UINT8', 'INT32'], ['UINT8']), (['UINT8', 'UINT16'], ['UINT8']), (['UINT8', 'UINT32'], ['UINT8'])], 'OpType.RgbToYuv': [], 'OpType.RNN': [], 'OpType.RoiAlign': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT8'], ['INT16']), (['INT8', 'UINT16'], ['INT8']), (['INT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT16'], ['UINT8']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.Round': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Rsqrt': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.SELU': [], 'OpType.SHRINK': [], 'OpType.ScatterElements': [(['BFLOAT16', 'INT16', 'BFLOAT16'], ['BFLOAT16']), (['BFLOAT16', 'INT32', 'BFLOAT16'], ['BFLOAT16']), (['BFLOAT16', 'INT8', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'INT16', 'FLOAT16'], ['FLOAT16']), (['FLOAT16', 'INT32', 'FLOAT16'], ['FLOAT16']), (['FLOAT16', 'INT8', 'FLOAT16'], ['FLOAT16']), (['INT16', 'INT16', 'INT16'], ['INT16']), (['INT16', 'INT32', 'INT16'], ['INT16']), (['INT16', 'INT8', 'INT16'], ['INT16']), (['INT8', 'INT16', 'INT8'], ['INT8']), (['INT8', 'INT32', 'INT8'], ['INT8']), (['INT8', 'INT8', 'INT8'], ['INT8']), (['UINT16', 'INT16', 'UINT16'], ['UINT16']), (['UINT16', 'INT32', 'UINT16'], ['UINT16']), (['UINT16', 'INT8', 'UINT16'], ['UINT16']), (['UINT8', 'INT16', 'UINT8'], ['UINT8']), (['UINT8', 'INT32', 'UINT8'], ['UINT8']), (['UINT8', 'INT8', 'UINT8'], ['UINT8'])], 'OpType.ScatterND': [(['BFLOAT16', 'UINT16', 'BFLOAT16'], ['BFLOAT16']), (['BFLOAT16', 'UINT32', 'BFLOAT16'], ['BFLOAT16']), (['BFLOAT16', 'UINT8', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'UINT16', 'FLOAT16'], ['FLOAT16']), (['FLOAT16', 'UINT32', 'FLOAT16'], ['FLOAT16']), (['FLOAT16', 'UINT8', 'FLOAT16'], ['FLOAT16']), (['INT16', 'UINT16', 'INT16'], ['INT16']), (['INT16', 'UINT32', 'INT16'], ['INT16']), (['INT16', 'UINT8', 'INT16'], ['INT16']), (['INT8', 'UINT16', 'INT8'], ['INT8']), (['INT8', 'UINT32', 'INT8'], ['INT8']), (['INT8', 'UINT8', 'INT8'], ['INT8']), (['UINT8', 'UINT16', 'UINT8'], ['UINT8']), (['UINT8', 'UINT32', 'UINT8'], ['UINT8']), (['UINT8', 'UINT8', 'UINT8'], ['UINT8'])], 'OpType.SegmentReduce': [(['INT8', 'UINT16'], ['INT8'])], 'OpType.Sigmoid': [], 'OpType.Sign': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Silu': [], 'OpType.Sine': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Sinh': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Slice': [(['BFLOAT16', 'INT32', 'INT32', 'INT32', 'INT32'], ['BFLOAT16', 'INT32']), (['BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'INT32', 'INT32', 'INT32', 'INT32'], ['FLOAT16', 'INT32']), (['FLOAT16'], ['FLOAT16']), (['INT16', 'INT32', 'INT32', 'INT32', 'INT32'], ['INT16', 'INT32']), (['INT16'], ['INT16']), (['INT8', 'INT32', 'INT32', 'INT32', 'INT32'], ['INT8', 'INT32']), (['INT8'], ['INT8']), (['UINT16', 'INT32', 'INT32', 'INT32', 'INT32'], ['UINT16', 'INT32']), (['UINT16'], ['UINT16']), (['UINT8', 'INT32', 'INT32', 'INT32', 'INT32'], ['UINT8', 'INT32']), (['UINT8'], ['UINT8'])], 'OpType.Softmax': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Softplus': [], 'OpType.Softsign': [], 'OpType.Sort': [], 'OpType.SpaceToBatch': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.SpaceToDepth': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.Split': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.Sqrt': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Square': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.SquaredDifference': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT16']), (['INT16', 'INT16'], ['UINT8']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT16'], ['INT8']), (['INT16', 'UINT16'], ['UINT16']), (['INT16', 'UINT16'], ['UINT8']), (['INT8', 'INT8'], ['INT16']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT16']), (['INT8', 'INT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT16']), (['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT8'], ['UINT16']), (['INT8', 'UINT8'], ['UINT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT16'], ['UINT8']), (['UINT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['INT8']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT16'], ['UINT8']), (['UINT8', 'INT8'], ['INT16']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT16']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT16']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT16']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.Squeeze': [], 'OpType.StridedSlice': [], 'OpType.Sub': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16']), (['INT16', 'INT16'], ['INT16']), (['INT16', 'INT16'], ['INT8']), (['INT16', 'INT16'], ['UINT16']), (['INT16', 'INT16'], ['UINT8']), (['INT16', 'UINT16'], ['INT16']), (['INT16', 'UINT16'], ['INT8']), (['INT16', 'UINT16'], ['UINT16']), (['INT16', 'UINT16'], ['UINT8']), (['INT32', 'INT32'], ['INT32']), (['INT8', 'INT8'], ['INT16']), (['INT8', 'INT8'], ['INT8']), (['INT8', 'INT8'], ['UINT16']), (['INT8', 'INT8'], ['UINT8']), (['INT8', 'UINT8'], ['INT16']), (['INT8', 'UINT8'], ['INT8']), (['INT8', 'UINT8'], ['UINT16']), (['INT8', 'UINT8'], ['UINT8']), (['UINT16', 'INT16'], ['INT16']), (['UINT16', 'INT16'], ['INT8']), (['UINT16', 'INT16'], ['UINT16']), (['UINT16', 'INT16'], ['UINT8']), (['UINT16', 'UINT16'], ['INT16']), (['UINT16', 'UINT16'], ['INT8']), (['UINT16', 'UINT16'], ['UINT16']), (['UINT16', 'UINT16'], ['UINT8']), (['UINT32', 'UINT32'], ['UINT32']), (['UINT8', 'INT8'], ['INT16']), (['UINT8', 'INT8'], ['INT8']), (['UINT8', 'INT8'], ['UINT16']), (['UINT8', 'INT8'], ['UINT8']), (['UINT8', 'UINT8'], ['INT16']), (['UINT8', 'UINT8'], ['INT8']), (['UINT8', 'UINT8'], ['UINT16']), (['UINT8', 'UINT8'], ['UINT8'])], 'OpType.Swish': [], 'OpType.THRESHOLDEDRELU': [], 'OpType.Tan': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT16'], ['INT16']), (['UINT16'], ['UINT16']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Tanh': [], 'OpType.Tile': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['FLOAT32'], ['FLOAT32']), (['INT16'], ['INT16']), (['INT32'], ['INT32']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT32'], ['UINT32']), (['UINT8'], ['UINT8'])], 'OpType.TopK': [(['BFLOAT16', 'INT32'], ['BFLOAT16', 'INT32']), (['BFLOAT16', 'INT32'], ['BFLOAT16', 'UINT16']), (['BFLOAT16', 'INT32'], ['BFLOAT16', 'UINT32']), (['BFLOAT16', 'UINT32'], ['BFLOAT16', 'INT32']), (['BFLOAT16', 'UINT32'], ['BFLOAT16', 'UINT16']), (['BFLOAT16', 'UINT32'], ['BFLOAT16', 'UINT32']), (['BFLOAT16'], ['BFLOAT16', 'INT32']), (['BFLOAT16'], ['BFLOAT16', 'UINT16']), (['BFLOAT16'], ['BFLOAT16', 'UINT32']), (['FLOAT16', 'INT32'], ['FLOAT16', 'INT32']), (['FLOAT16', 'INT32'], ['FLOAT16', 'UINT16']), (['FLOAT16', 'INT32'], ['FLOAT16', 'UINT32']), (['FLOAT16', 'UINT32'], ['FLOAT16', 'INT32']), (['FLOAT16', 'UINT32'], ['FLOAT16', 'UINT16']), (['FLOAT16', 'UINT32'], ['FLOAT16', 'UINT32']), (['FLOAT16'], ['FLOAT16', 'INT32']), (['FLOAT16'], ['FLOAT16', 'UINT16']), (['FLOAT16'], ['FLOAT16', 'UINT32']), (['INT16', 'INT32'], ['INT16', 'UINT16']), (['INT16', 'INT32'], ['INT16', 'UINT32']), (['INT16', 'UINT32'], ['INT16', 'UINT16']), (['INT16', 'UINT32'], ['INT16', 'UINT32']), (['INT16'], ['INT16', 'UINT16']), (['INT16'], ['INT16', 'UINT32']), (['INT8', 'INT32'], ['INT8', 'UINT16']), (['INT8', 'INT32'], ['INT8', 'UINT32']), (['INT8', 'UINT32'], ['INT8', 'UINT16']), (['INT8', 'UINT32'], ['INT8', 'UINT32']), (['INT8'], ['INT8', 'UINT16']), (['INT8'], ['INT8', 'UINT32']), (['UINT16', 'INT32'], ['UINT16', 'UINT16']), (['UINT16', 'INT32'], ['UINT16', 'UINT32']), (['UINT16', 'UINT32'], ['UINT16', 'UINT16']), (['UINT16', 'UINT32'], ['UINT16', 'UINT32']), (['UINT16'], ['UINT16', 'UINT16']), (['UINT16'], ['UINT16', 'UINT32']), (['UINT8', 'INT32'], ['UINT8', 'UINT16']), (['UINT8', 'INT32'], ['UINT8', 'UINT32']), (['UINT8', 'UINT32'], ['UINT8', 'UINT16']), (['UINT8', 'UINT32'], ['UINT8', 'UINT32']), (['UINT8'], ['UINT8', 'UINT16']), (['UINT8'], ['UINT8', 'UINT32'])], 'OpType.Transpose': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT32'], ['INT32']), (['INT8'], ['INT8']), (['UINT16'], ['UINT16']), (['UINT32'], ['UINT32']), (['UINT8'], ['UINT8'])], 'OpType.UpsampleByIndex': [(['INT16', 'INT32'], ['INT16']), (['INT8', 'INT32'], ['INT8']), (['UINT16', 'INT32'], ['UINT16']), (['UINT8', 'INT32'], ['UINT8'])], 'OpType.Where': [(['INT16', 'INT16', 'INT16'], ['INT16']), (['INT16'], ['INT16']), (['INT16'], ['INT8']), (['INT16'], ['UINT8']), (['INT8', 'INT8', 'INT8'], ['INT8']), (['INT8'], ['INT8']), (['UINT16', 'UINT16', 'UINT16'], ['UINT16']), (['UINT16'], ['UINT16']), (['UINT8', 'UINT8', 'UINT8'], ['UINT8']), (['UINT8'], ['UINT8'])], 'OpType.YuvToRgb': [], 'OpType.ZeroFraction': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['UINT16']), (['INT8'], ['UINT8']), (['UINT16'], ['UINT16']), (['UINT8'], ['UINT8'])], 'OpType.DivMod': [(['BFLOAT16', 'BFLOAT16'], ['BFLOAT16', 'BFLOAT16']), (['FLOAT16', 'FLOAT16'], ['FLOAT16', 'FLOAT16']), (['INT16', 'INT16'], ['INT16', 'INT16']), (['INT32', 'INT32'], ['INT32', 'INT32']), (['INT8', 'INT8'], ['INT8', 'INT8']), (['UINT16', 'UINT16'], ['UINT16', 'UINT16']), (['UINT32', 'UINT32'], ['UINT32', 'UINT32']), (['UINT8', 'UINT8'], ['UINT8', 'UINT8'])], 'OpType.NonZero': [(['BFLOAT16'], ['INT32', 'UINT32']), (['BFLOAT16'], ['UINT16', 'UINT32']), (['BFLOAT16'], ['UINT32', 'UINT32']), (['BFLOAT16'], ['UINT8', 'UINT32']), (['FLOAT16'], ['INT32', 'UINT32']), (['FLOAT16'], ['UINT16', 'UINT32']), (['FLOAT16'], ['UINT32', 'UINT32']), (['FLOAT16'], ['UINT8', 'UINT32']), (['INT16'], ['UINT16', 'UINT32']), (['INT16'], ['UINT32', 'UINT32']), (['INT16'], ['UINT8', 'UINT32']), (['INT8'], ['UINT16', 'UINT32']), (['INT8'], ['UINT32', 'UINT32']), (['INT8'], ['UINT8', 'UINT32']), (['UINT16'], ['UINT16', 'UINT32']), (['UINT16'], ['UINT32', 'UINT32']), (['UINT16'], ['UINT8', 'UINT32']), (['UINT8'], ['UINT16', 'UINT32']), (['UINT8'], ['UINT32', 'UINT32']), (['UINT8'], ['UINT8', 'UINT32'])], 'OpType.SlotUpdate': [], 'OpType.QueryRebatch': [], 'OpType.RMSNorm': [(['BFLOAT16'], ['BFLOAT16']), (['FLOAT16'], ['FLOAT16']), (['INT16'], ['INT16']), (['INT16'], ['UINT16']), (['INT8'], ['INT8']), (['INT8'], ['UINT8']), (['UINT8'], ['INT8']), (['UINT8'], ['UINT8'])], 'OpType.Logit': []}
    dtype_specs = []
    if str(ntype) in static_all_op_dtype_spec.keys():
        from AIPUBuilder.Optimizer.utils.dtype_utils import str2dtype
        for c_spec in static_all_op_dtype_spec[str(ntype)]:
            spec = OpDtypeSpec()
            spec.in_dtypes = []
            spec.out_dtypes = []
            for s in c_spec[0]:
                spec.in_dtypes.append(str2dtype(s))
            for s in c_spec[1]:
                spec.out_dtypes.append(str2dtype(s))
            dtype_specs.append(spec)
    return dtype_specs
#################################################################################
