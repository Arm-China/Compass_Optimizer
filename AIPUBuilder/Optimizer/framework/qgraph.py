# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3
from AIPUBuilder.Optimizer.framework.pycore.pygraph import PyGraph

__all__ = [
    'graph_inference',
    'QuantizeGraph'
]


def graph_inference(graph, g_forward, dataloader, metrics, with_float=False, max_batches=0, disable_tqdm=False):
    import sys
    import torch
    from AIPUBuilder.Optimizer.framework.pycore.pytensor import opt_use_cuda
    from AIPUBuilder.Optimizer.utils.quant_tool_utils import linear_dequantize
    from AIPUBuilder.Optimizer.logger import tqdm
    desc = 'float metric batch' if with_float else 'quant metric batch'
    graph.current_batch_size = dataloader.batch_size
    current_batch_idx = 0
    with tqdm(dataloader, desc=desc, file=sys.stdout, disable=disable_tqdm, consumer=graph) as pbar:
        for i, sample in enumerate(pbar):
            graph.current_batch_idx = current_batch_idx
            current_batch_idx += 1
            if current_batch_idx * dataloader.batch_size > len(dataloader.dataset):
                graph.current_batch_size = len(dataloader.dataset) - (current_batch_idx - 1) * dataloader.batch_size
            inp, target = sample
            if opt_use_cuda():
                if isinstance(target, dict):
                    target = {key: target[key].cuda() if isinstance(target[key], torch.Tensor) else target[key]
                              for key in target}
                elif isinstance(target, (list, tuple)):
                    target = [t.cuda() if isinstance(t, torch.Tensor) else t for t in target]
                else:
                    target = target.cuda() if isinstance(target, torch.Tensor) else target

                if isinstance(inp, dict):
                    inp = {key: inp[key].cuda() for key in inp}
                elif isinstance(inp, (list, tuple)):
                    inp = [ii.cuda() for ii in inp]
                else:
                    inp = inp.cuda()
            out = g_forward(inp)
            # dequantize quantized forward's output tensors for consistently call metirc functions
            prediction = []
            for t in out:
                if with_float:
                    prediction.append(t.betensor)
                else:
                    if t.debug_flag or (t.pnode is not None and t.pnode.get_param('unquantifiable', optional=True, default_value=False)):
                        dtb = t.betensor
                    else:
                        dtb = linear_dequantize(t.betensor, t.scale, t.zerop)
                    prediction.append(dtb)
            for metric in metrics:
                metric(prediction, target)
            if max_batches > 0 and current_batch_idx >= max_batches:
                break
        pbar.refresh()
    if opt_use_cuda():
        torch.cuda.empty_cache()


class QuantizeGraph(PyGraph):
    def __init__(self, name="unamed"):
        super().__init__(name)
        self.quantgraph = None
        self.constants_statisticed = False

    def clone(self):
        clone_graph = super().clone()
        clone_graph.constants_statisticed = self.constants_statisticed
        return clone_graph

    def feed_inputs_data(self, inputs_data):
        from AIPUBuilder.Optimizer.framework.pycore.pytensor import opt_use_cuda
        from AIPUBuilder.Optimizer.framework import PyTensor
        feed_data = inputs_data
        if opt_use_cuda():
            if isinstance(feed_data, dict):
                feed_data = {key: PyTensor('tmp', feed_data[key]).betensor.cuda() for key in feed_data}
            elif isinstance(feed_data, (list, tuple)):
                feed_data = [PyTensor('tmp', ii).betensor.cuda() for ii in feed_data]
            else:
                feed_data = PyTensor('tmp', feed_data).betensor.cuda()
        data = feed_data
        if len(self.input_tensors) == 1 and not isinstance(feed_data, list):
            data = [feed_data, ]
        for inp, d in zip(self.input_tensors, data):
            inp.betensor = PyTensor('tmp', d).betensor
            inp.betensor = inp.betensor.float()

    def statistic(self, inputs, config):
        import sys
        import re
        from AIPUBuilder.Optimizer.framework.pycore.pytensor import TensorShape, PyTensor
        from AIPUBuilder.Optimizer.utils import QuantMode
        from AIPUBuilder.Optimizer.config import CalibrationStrategyField
        from AIPUBuilder.Optimizer.logger import tqdm, OPT_DEBUG, OPT_ERROR

        time_saving_mode = not config.save_statistic_info
        trim_inf = {}
        for n in self.nodes:
            dv = ((float('-inf'), float('inf')), '')
            tcmd = [x for x in re.split(
                r',|\(|\)', n.attrs['trim_infinity_before_statistic'].strip()) if x.lower().strip()]
            trim_inf[n] = dv if len(tcmd) < 3 else ((float(tcmd[1]), float(tcmd[2])), str(tcmd[0]))

        if not self.constants_statisticed:
            with tqdm(total=len(self.nodes), desc='statistic weights and biases', file=sys.stdout, leave=False) as pbar:
                for n in self.nodes:
                    running_statistic_momentum = n.attrs["running_statistic_momentum"]
                    histc_bins = n.attrs["histc_bins"]
                    statistic_std_mean = True
                    cstrategy = n.get_attrs('q_strategy_weight')
                    qmethod_wht = n.get_attrs('q_mode_weight')
                    if not n.quantized:
                        for v in list(n.constants.values()) + n.get_attrs('subgraph_constants', optional=True, default_value=[]):
                            key_axis = v.key_axis if v.ir_shape != TensorShape([]) else None
                            if time_saving_mode:
                                if not (QuantMode.is_per_channel(qmethod_wht) or n.get_param('group', optional=True, default_value=1) > 1):
                                    key_axis = None
                                r = CalibrationStrategyField._need_statistic_info(cstrategy)
                                histc_bins = None if not r['histc'] else histc_bins
                                statistic_std_mean = False if not r['std_mean'] else statistic_std_mean
                            v.statistic(running_statistic_momentum, key_axis=key_axis, key_axis_g=v.key_axis_g,
                                        histc_bins=histc_bins, statistic_std_mean=statistic_std_mean,
                                        trim_infinity=trim_inf[n],
                                        reset=True)
                    pbar.update(1)
                pbar.refresh()
        self.constants_statisticed = True
        self.reset_edge_tensors_ref_count()
        tz = PyTensor('null').betensor
        self.feed_inputs_data(inputs)
        for n in self.nodes:
            n.forward()
            running_statistic_momentum = n.attrs["running_statistic_momentum"]
            histc_bins = n.attrs["histc_bins"]
            statistic_std_mean = True
            astrategy = n.get_attrs('q_strategy_activation')
            qmethod_act = n.get_attrs('q_mode_activation')
            if not n.quantized:
                r = CalibrationStrategyField._need_statistic_info(astrategy)
                if time_saving_mode:
                    histc_bins = None if not r['histc'] else histc_bins
                    statistic_std_mean = False if not r['std_mean'] else statistic_std_mean
                try:
                    for o in n.outputs:
                        key_axis = o.key_axis if o.ir_shape != TensorShape([]) else None
                        o.statistic(running_statistic_momentum, key_axis=key_axis, key_axis_g=o.key_axis_g,
                                    histc_bins=histc_bins, statistic_std_mean=statistic_std_mean,
                                    trim_infinity=trim_inf[n],
                                    reset=not self.current_batch_idx)
                    for p in n.placeholders:
                        p_key_axis = None if not QuantMode.is_per_channel(qmethod_act) else p.key_axis
                        p.statistic(running_statistic_momentum, key_axis=p_key_axis, key_axis_g=p.key_axis_g,
                                    histc_bins=histc_bins, statistic_std_mean=statistic_std_mean,
                                    trim_infinity=trim_inf[n],
                                    reset=not self.current_batch_idx)
                except Exception as e:
                    OPT_ERROR(f"{n}, {o} statistic failed")
                    raise e
            for pld in n.placeholders:
                del pld.betensor
                pld.betensor = tz
        for n in self.nodes:
            for t in n.outputs:
                if t not in self.output_tensors:
                    del t.betensor
                    t.betensor = tz
        self.reset_edge_tensors_ref_count()

    def save_statistic_info(self, statistic_info_fname):
        import pickle
        import torch
        from AIPUBuilder.Optimizer.logger import OPT_WARN
        from AIPUBuilder.Optimizer.utils.files_utils import make_path

        def get_value(t_param):
            return t_param.cpu().contiguous().numpy() if isinstance(t_param, torch.Tensor) else t_param

        statistic_info = {}
        for n in self.nodes:
            statistic_info[n.name] = {}
            for o in n.outputs:
                statistic_info[n.name][o.name] = {
                    "extrema_min": get_value(o.extrema_min),
                    "extrema_max": get_value(o.extrema_max),
                    "running_min": get_value(o.running_min),
                    "running_max": get_value(o.running_max),
                    "running_mean": get_value(o.running_mean),
                    "running_std": get_value(o.running_std),
                    "running_mad": get_value(o.running_mad),
                    "running_histc": get_value(o.running_histc),
                }
            for _, v in n.constants.items():
                statistic_info[n.name][v.name] = {
                    "extrema_min": get_value(v.extrema_min),
                    "extrema_max": get_value(v.extrema_max),
                    "running_min": get_value(v.running_min),
                    "running_max": get_value(v.running_max),
                    "running_mean": get_value(v.running_mean),
                    "running_std": get_value(v.running_std),
                    "running_mad": get_value(v.running_mad),
                    "running_histc": get_value(v.running_histc),
                    "extrema_min_key_axis": get_value(v.extrema_min_key_axis),
                    "extrema_max_key_axis": get_value(v.extrema_max_key_axis),
                    "running_min_key_axis": get_value(v.running_min_key_axis),
                    "running_max_key_axis": get_value(v.running_max_key_axis),
                    "running_mean_key_axis": get_value(v.running_mean_key_axis),
                    "running_std_key_axis": get_value(v.running_std_key_axis),
                    "running_mad_key_axis": get_value(v.running_mad_key_axis),
                    "running_histc_key_axis": get_value(v.running_histc_key_axis),
                }
            for p in n.placeholders:
                statistic_info[n.name][p.name] = {
                    "extrema_min": get_value(p.extrema_min),
                    "extrema_max": get_value(p.extrema_max),
                    "running_min": get_value(p.running_min),
                    "running_max": get_value(p.running_max),
                    "running_mean": get_value(p.running_mean),
                    "running_std": get_value(p.running_std),
                    "running_mad": get_value(p.running_mad),
                    "running_histc": get_value(p.running_histc),
                }
        statistic_info_fname = make_path(statistic_info_fname)
        try:
            # np.save(statistic_info_fname, statistic_info)
            with open(statistic_info_fname, 'wb') as fw:
                pickle.dump(statistic_info, fw)
        except Exception as e:
            OPT_WARN(f"Optimizer saves the statistic file failed, because {e}")

    def load_statistic_info(self, statistic_info_fname, ignore_missing=False):
        import numpy as np
        import torch
        from AIPUBuilder.Optimizer.logger import OPT_FATAL, OPT_INFO, OPT_ERROR, OPT_WARN
        from AIPUBuilder.Optimizer.framework.pycore.pytensor import opt_use_cuda
        # statistic_info = np.load(statistic_info_fname, allow_pickle=True).item()
        statistic_info = np.load(statistic_info_fname, allow_pickle=True)
        if isinstance(statistic_info, np.ndarray):
            statistic_info = statistic_info.item()

        def query_property(node_name, tensor_name, property):
            if not node_name in statistic_info:
                OPT_FATAL("can not find node '%s' in file %s, please update statistic_file by regenerating it." %
                          (node_name, statistic_info_fname))
                return None
            if not tensor_name in statistic_info[node_name]:
                OPT_FATAL("can not find tensor '%s' in file %s, please update statistic_file by regenerating it." %
                          (tensor_name, statistic_info_fname))
                return None
            if not property in statistic_info[node_name][tensor_name]:
                OPT_FATAL("can not find '%s' of tensor '%s' in file %s, please update statistic_file by regenerating it." %
                          (property, tensor_name, statistic_info_fname))
                return None
            value = statistic_info[node_name][tensor_name][property]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if isinstance(value, torch.Tensor):
                value = value.cuda() if opt_use_cuda() else value.cpu()
            return value

        def copy_tensor_stat(src, tgt):
            src.extrema_min = tgt.extrema_min
            src.extrema_max = tgt.extrema_max
            src.running_min = tgt.running_min
            src.running_max = tgt.running_max
            src.running_mean = tgt.running_mean
            src.running_std = tgt.running_std
            src.running_mad = tgt.running_mad
            src.running_histc = tgt.running_histc
        for n in self.nodes:
            if not n.name in statistic_info and ignore_missing:
                OPT_WARN("can not find node '%s' in file %s, please update statistic_file by regenerating it.\
                     Trying to fill values by its parent nodes..." %
                         (n.name, statistic_info_fname))
                # Try find one valid parent input and apply to itself as well as other unassigned parent inputs
                if len(n.parents) > 0:
                    tgt = None
                    for inp in n.inputs:
                        if inp.running_histc is not None:
                            tgt = inp
                            break
                    if tgt is not None:
                        for oup in n.outputs:
                            copy_tensor_stat(oup, tgt)
                        for inp in n.inputs:
                            if inp.running_histc is None:
                                copy_tensor_stat(inp, tgt)
                continue
            for o in n.outputs:
                o.extrema_min = query_property(n.name,   o.name, "extrema_min")
                o.extrema_max = query_property(n.name,   o.name, "extrema_max")
                o.running_min = query_property(n.name,   o.name, "running_min")
                o.running_max = query_property(n.name,   o.name, "running_max")
                o.running_mean = query_property(n.name,  o.name, "running_mean")
                o.running_std = query_property(n.name,   o.name, "running_std")
                o.running_mad = query_property(n.name,   o.name, "running_mad")
                o.running_histc = query_property(n.name, o.name, "running_histc")
            for _, v in n.constants.items():
                v.extrema_min = query_property(n.name,   v.name, "extrema_min")
                v.extrema_max = query_property(n.name,   v.name, "extrema_max")
                v.running_min = query_property(n.name,   v.name, "running_min")
                v.running_max = query_property(n.name,   v.name, "running_max")
                v.running_mean = query_property(n.name,  v.name, "running_mean")
                v.running_std = query_property(n.name,   v.name, "running_std")
                v.running_mad = query_property(n.name,   v.name, "running_mad")
                v.running_histc = query_property(n.name, v.name, "running_histc")
                v.extrema_min_key_axis = query_property(n.name,   v.name, "extrema_min_key_axis")
                v.extrema_max_key_axis = query_property(n.name,   v.name, "extrema_max_key_axis")
                v.running_min_key_axis = query_property(n.name,   v.name, "running_min_key_axis")
                v.running_max_key_axis = query_property(n.name,   v.name, "running_max_key_axis")
                v.running_mean_key_axis = query_property(n.name,  v.name, "running_mean_key_axis")
                v.running_std_key_axis = query_property(n.name,   v.name, "running_std_key_axis")
                v.running_mad_key_axis = query_property(n.name,   v.name, "running_mad_key_axis")
                v.running_histc_key_axis = query_property(n.name, v.name, "running_histc_key_axis")
            for p in n.placeholders:
                p.extrema_min = query_property(n.name,   p.name, "extrema_min")
                p.extrema_max = query_property(n.name,   p.name, "extrema_max")
                p.running_min = query_property(n.name,   p.name, "running_min")
                p.running_max = query_property(n.name,   p.name, "running_max")
                p.running_mean = query_property(n.name,  p.name, "running_mean")
                p.running_std = query_property(n.name,   p.name, "running_std")
                p.running_mad = query_property(n.name,   p.name, "running_mad")
                p.running_histc = query_property(n.name, p.name, "running_histc")
        OPT_INFO('Succesfully loaded statistic info from file: ' + statistic_info_fname)
        return True

    def set_fp_tensor_after_update_quantization_attr(self):
        import sys
        from AIPUBuilder.Optimizer.logger import tqdm
        from AIPUBuilder.Optimizer.utils import str2dtype, is_float
        from AIPUBuilder.Optimizer.framework import Dtype, OpType
        with tqdm(total=len(self.nodes), desc='update_fp_tensor_quantization_attrs', file=sys.stdout, leave=False) as pbar:
            for n in self.nodes:
                if n.params['unquantifiable']:
                    if n.type == OpType.Quantize:
                        continue
                    fd = (
                        n.attrs["trigger_float_op"].name
                        if isinstance(n.attrs["trigger_float_op"], Dtype)
                        else str(n.attrs["trigger_float_op"]).lower().strip()
                    )
                    o_dtype = str2dtype(fd)
                    for ot in n.outputs:
                        if is_float(ot.ir_dtype):
                            ot.dtype = o_dtype
                        else:
                            ot.dtype = ot.ir_dtype
                pbar.update(1)
            pbar.refresh()

    def set_tensor_quantization_attrs(self):
        import sys
        from AIPUBuilder.Optimizer.logger import tqdm
        from AIPUBuilder.Optimizer.framework import OpType

        # subgraph is not suitable for quantization
        if self.root_graph != None:
            return
        with tqdm(total=len(self.nodes), desc='update_tensor_quantization_attrs', file=sys.stdout, leave=False) as pbar:
            for n in self.nodes:
                if n.type in [OpType.If, OpType.Loop]:
                    continue
                qn = n.clone(n.name+"_clone_")
                qn.params['unquantifiable'] = False
                qn.quantize()
                for i, t in enumerate(n.outputs):
                    tc = qn.outputs[i]
                    t.clone_qinfo(tc)
                for i, t in enumerate(n.placeholders):
                    tc = qn.placeholders[i]
                    t.clone_qinfo(tc)
                for k, t in n.constants.items():
                    if k in qn.constants.keys():
                        tc = qn.constants[k]
                        t.clone_qinfo(tc)
                pbar.update(1)
            pbar.refresh()

    def clear_tensor_quantization_attrs(self):
        from AIPUBuilder.Optimizer.utils.dtype_utils import torch_type2dtype
        for n in self.nodes:
            for i, t in enumerate(n.outputs):
                t.scale = 1.0
                t.zerop = 0
                t.qbits = None
                t.dtype = t.ir_dtype
                t.qmin = None
                t.qmax = None
                t.qinvariant = None
            for i, t in enumerate(n.placeholders):
                t.scale = 1.0
                t.zerop = 0
                t.qbits = None
                t.dtype = torch_type2dtype(t.betensor.dtype)
                t.qmin = None
                t.qmax = None
                t.qinvariant = None
            for k, t in n.constants.items():
                if not ((k == 'weights') and all(sname in n.constants.keys() for sname in ['scale0', 'scale1'])):
                    t.scale = 1.0
                    t.zerop = 0
                t.qbits = None
                t.dtype = t.ir_dtype
                t.qmin = None
                t.qmax = None
                t.qinvariant = None

    def quantize(self, disable_pbar=False):
        import sys
        from AIPUBuilder.Optimizer.logger import tqdm, OPT_WARN, OPT_DEBUG
        from AIPUBuilder.Optimizer.framework import OpType, Dtype, PyTensor, PyNode
        from AIPUBuilder.Optimizer.utils import (is_float, is_signed, str2dtype, QuantMode,
                                                 get_linear_quant_params_from_tensor,
                                                 torch_type2dtype, dtype2bits,
                                                 linear_quantize_clip)

        if self.quantgraph is None:
            self.quantgraph = self.clone()

        for _, sg in self.quantgraph.subgraph_map.items():
            sg.quantgraph = sg
            sg.quantize(disable_pbar=True)
            sg.quantgraph = None

        self.quantgraph.set_tensor_quantization_attrs()  # pylint: disable=no-member

        # record the map between quantized node's name and source node object pointer
        qnmap = {}
        for qn, n in zip(self.quantgraph.nodes, self.nodes):
            qnmap[qn.name] = n
        for qn in self.quantgraph.nodes:
            qn.attrs['map_to_original_node'] = qnmap

        with tqdm(total=len(self.quantgraph.nodes), desc='quantize each layer', file=sys.stdout, leave=True, disable=disable_pbar) as pbar:
            for n in self.quantgraph.nodes:
                if not n.quantized:
                    n.quantize()
                    # n.quantized = True
                pbar.update(1)
            pbar.refresh()
        new_cast_edges_in_qg = {}
        for gn, qn in zip(self.nodes, self.quantgraph.nodes):
            fd = qn.attrs['trigger_float_op'].name if isinstance(qn.attrs['trigger_float_op'], Dtype) \
                else str(qn.attrs['trigger_float_op']).lower().strip()
            if fd != 'disable' and qn.params['unquantifiable']:
                if qn.type == OpType.Quantize:
                    if not is_float(qn.inputs[0].ir_dtype):
                        qn.params.clear()
                        qn.params['to_dtype'] = qn.outputs[0].dtype
                        qn.params['ignore_scale_zp'] = True
                        qn.params['clip_mode'] = 'TRUNCATION'
                        qn.params['unquantifiable'] = True
                        qn.type = OpType.Cast
                    else:
                        qn.set_ir_field('quantize_scale', qn.outputs[0].scale, Dtype.FP32)
                        qn.set_ir_field('quantize_zp', qn.outputs[0].zerop, Dtype.INT32)
                if qn.type == OpType.DeQuantize:
                    if not is_float(qn.outputs[0].ir_dtype):
                        qn.params.clear()
                        qn.params['to_dtype'] = qn.outputs[0].ir_dtype
                        qn.params['ignore_scale_zp'] = True
                        qn.params['clip_mode'] = 'TRUNCATION'
                        qn.params['unquantifiable'] = True
                        qn.type = OpType.Cast
                    else:
                        qn.set_ir_field('quantize_scale', qn.inputs[0].scale, Dtype.FP32)
                        qn.set_ir_field('quantize_zp', qn.inputs[0].zerop, Dtype.INT32)

                if qn.type != OpType.Quantize:
                    for i, t in enumerate(qn.outputs):
                        t.scale = 1.0
                        t.zerop = 0
                        t.qbits = None
                        t.dtype = t.ir_dtype
                        t.qmin = None
                        t.qmax = None
                        t.qinvariant = None
                    for i, t in enumerate(qn.placeholders):
                        t.scale = 1.0
                        t.zerop = 0
                        t.qbits = None
                        t.dtype = torch_type2dtype(t.betensor.dtype)
                        t.qmin = None
                        t.qmax = None
                        t.qinvariant = None
                    for k, t in qn.constants.items():
                        if not ((k == 'weights') and all(sname in qn.constants.keys() for sname in ['scale0', 'scale1'])):
                            t.scale = 1.0
                            t.zerop = 0
                        t.qbits = None
                        t.dtype = t.ir_dtype
                        t.qmin = None
                        t.qmax = None
                        t.qinvariant = None
                    if all(sname in qn.constants.keys() for sname in ['weights', 'scale0', 'scale1']):
                        # print("pop scale0")
                        qn.constants.pop('scale0')
                        qn.constants.pop('scale1')
                        qn.constants.pop('zp0')
                        # v = qn.constants['weights']
                        # scales = v.scale
                        # zps = v.zerop
                        # ir_shape_len = len(v.ir_shape)
                        # if ir_shape_len and ir_shape_len - 1 >= v.key_axis:
                        #     channel_num = v.ir_shape[v.key_axis]
                        #     if scales.numel() == zps.numel() and scales.numel() > 1 and scales.numel() > channel_num:
                        #         ic = qn.inputs[0].ir_shape[-1]
                        #         group_size = ic // (scales.numel() // channel_num)
                        #         qn.params['weight_block_size'] = group_size

                o_dtype = str2dtype(fd)
                o_int_index = []
                i_int_index = []
                inputs_dtype = []
                outputs_dtype = []
                for idx, ot in enumerate(qn.outputs):
                    if is_float(ot.dtype) and dtype2bits(ot.dtype) >= 16:
                        ot.dtype = o_dtype
                    else:
                        o_int_index.append(idx)
                    outputs_dtype.append(ot.dtype)
                for idx, it in enumerate(qn.inputs):
                    if not is_float(it.dtype):
                        i_int_index.append(idx)
                    inputs_dtype.append(it.dtype)

                if self.quantgraph.op_need_cast_dtypes_for_lib is not None and qn in self.quantgraph.op_need_cast_dtypes_for_lib:
                    import itertools

                    if qn.type in [OpType.Quantize, OpType.DeQuantize]:
                        continue
                    ############In the trigger_float_op case, convert the float input type to float type supported by lib###########
                    for gnt, qnt in zip(gn.inputs, qn.inputs):
                        if is_float(qnt.dtype) and dtype2bits(qnt.dtype) >= 16 and qnt.dtype != o_dtype:
                            qnkey = (qnt, qn, gnt, gn)
                            new_cast_edges_in_qg[qnkey] = [qn.attrs, qnt.dtype, o_dtype]

                    ############In the trigger_float_op case, convert the int output type to int type supported by lib###########

                    def whether_matched(lib_dtype_spec, inputs_dtype, outputs_dtype):
                        for spec in lib_dtype_spec:
                            if outputs_dtype == spec.out_dtypes and inputs_dtype == spec.in_dtypes:
                                return True
                        return False

                    lib_dtype_spec = qn.get_lib_dtype_spec()
                    matched = len(lib_dtype_spec) == 0 or whether_matched(lib_dtype_spec, inputs_dtype, outputs_dtype)

                    if not matched:
                        candidate_output_dtypes = []

                        def get_lower_dtypes(dtype):
                            from AIPUBuilder.Optimizer.utils.dtype_utils import bits2dtype, dtype2bits, is_signed
                            lower_dtypes = []
                            candidate_bits = []
                            bits = dtype2bits(dtype)
                            signed = is_signed(dtype)
                            if not signed:
                                candidate_bits.append(bits)
                            while bits > 8:
                                bits = bits // 2
                                candidate_bits.append(bits)
                            for b, s in itertools.product(candidate_bits, [True, False]):
                                new_dtype = bits2dtype(b, s)
                                lower_dtypes.append(new_dtype)
                            return lower_dtypes

                        for int_index in o_int_index:
                            current_dtypes = outputs_dtype[int_index]
                            # such as candidate_output_dtypes = [['int8,int16'], ['int32']]
                            candidate_output_dtypes.append(get_lower_dtypes(current_dtypes))

                        for can_output_dtypes in itertools.product(*candidate_output_dtypes):
                            tmp_outputs_dtype = [xt for xt in outputs_dtype]
                            for o_idx, otype in enumerate(can_output_dtypes):
                                tmp_outputs_dtype[o_int_index[o_idx]] = otype
                            matched = whether_matched(lib_dtype_spec, inputs_dtype, tmp_outputs_dtype)
                            if matched:
                                for int_index in o_int_index:
                                    if qn.outputs[int_index].dtype != tmp_outputs_dtype[int_index]:
                                        qnt = qn.outputs[int_index]
                                        has_consumer = False
                                        for gcld, cld in zip(gn.children, qn.children):
                                            if qnt in cld.inputs:
                                                has_consumer = True
                                                qckey = (qnt, cld, gn.outputs[int_index], gcld)
                                                if qckey not in new_cast_edges_in_qg.keys():
                                                    new_cast_edges_in_qg[qckey] = [
                                                        qn.attrs, tmp_outputs_dtype[int_index], qnt.dtype]
                                                else:
                                                    new_cast_edges_in_qg[qckey].append(qnt.dtype)
                                        if not has_consumer:
                                            qnt.dtype = tmp_outputs_dtype[int_index]
                                break
                        if not matched:
                            best_candidate = None
                            best_cbits = 0
                            float_candidate = None
                            for spec in lib_dtype_spec:
                                sp_dt_list = spec.out_dtypes + spec.in_dtypes
                                if any([is_float(sdt) and sdt != o_dtype for sdt in sp_dt_list]):
                                    continue
                                cbits = 0
                                for int_index in o_int_index:
                                    if int_index >= len(spec.out_dtypes):
                                        continue
                                    sot = spec.out_dtypes[int_index]
                                    if not is_float(sot) and not (is_signed(outputs_dtype[int_index]) and not is_signed(sot)):
                                        cbits += dtype2bits(sot)
                                for int_index in i_int_index:
                                    if int_index >= len(spec.in_dtypes):
                                        continue
                                    sit = spec.in_dtypes[int_index]
                                    if not is_float(sit) and not (is_signed(inputs_dtype[int_index]) and not is_signed(sit)):
                                        cbits += dtype2bits(sit)
                                if cbits > best_cbits:
                                    best_cbits = cbits
                                    best_candidate = spec
                            if best_candidate is not None:
                                for int_index in o_int_index:
                                    if int_index >= len(best_candidate.out_dtypes):
                                        continue
                                    if qn.outputs[int_index].dtype != best_candidate.out_dtypes[int_index]:
                                        qnt = qn.outputs[int_index]
                                        has_consumer = False
                                        for gcld, cld in zip(gn.children, qn.children):
                                            if qnt in cld.inputs:
                                                has_consumer = True
                                                qckey = (qnt, cld, gn.outputs[int_index], gcld)
                                                if qckey not in new_cast_edges_in_qg.keys():
                                                    new_cast_edges_in_qg[qckey] = [
                                                        qn.attrs, best_candidate.out_dtypes[int_index], qnt.dtype]
                                                else:
                                                    new_cast_edges_in_qg[qckey].append(qnt.dtype)
                                        if not has_consumer:
                                            qnt.dtype = best_candidate.out_dtypes[int_index]
                                for int_index in i_int_index:
                                    if int_index >= len(best_candidate.in_dtypes):
                                        continue
                                    if qn.inputs[int_index].dtype != best_candidate.in_dtypes[int_index]:
                                        qnt = qn.inputs[int_index]
                                        qckey = (qnt, qn, gn.inputs[int_index], gn)
                                        if qckey not in new_cast_edges_in_qg.keys():
                                            new_cast_edges_in_qg[qckey] = [
                                                qn.attrs, qnt.dtype, best_candidate.in_dtypes[int_index]]
                                        else:
                                            new_cast_edges_in_qg[qckey].append(best_candidate.in_dtypes[int_index])

                if qn.type == OpType.Cast:
                    qn.params['to_dtype'] = qn.outputs[0].dtype

                if qn.type != OpType.Quantize and qn.type != OpType.DeQuantize:  # per-channel scale/zp would be in constants
                    for key, ct in qn.constants.items():
                        if is_float(ct.ir_dtype) and (not all(sname in qn.constants.keys() for sname in ['scale0', 'scale1'])):
                            ct.dtype = Dtype.FP32 if key in ['biases', 'negative_slope'] else o_dtype

                if "weights" in qn.constants.keys() and qn.get_attrs("weight_only_quantization", optional=True, default_value=False):
                    w = qn.constants["weights"]
                    q_mode_weight = qn.attrs["q_mode_weight"]
                    w.qbits = qn.attrs["q_bits_weight"]
                    w.qinvariant = False
                    # replace quantized weights if gptq optimization was applied
                    if 'gptq_weights' in qn.attrs:
                        bkey = qn.attrs['q_bits_weight']
                        if bkey in qn.attrs['gptq_weights'].keys():
                            OPT_DEBUG(f'gptq optimization was applied on {qn}')
                            w.betensor = qn.attrs['gptq_weights'][bkey]
                    if QuantMode.is_per_block(q_mode_weight):
                        w.block_size = qn.attrs["weight_block_size"]
                        from AIPUBuilder.Optimizer.features import statistic_and_calibration
                        wb = PyTensor('weight_blocks')
                        wb.betensor = w.betensor.reshape(-1, w.block_size)
                        wb.key_axis = 0
                        statistic_and_calibration(wb, qn.attrs, is_constant_tensor=True)
                        w.scale, w.zerop, w.qmin, w.qmax, w.dtype = get_linear_quant_params_from_tensor(
                            wb, QuantMode.to_per_channel(q_mode_weight), w.qbits, is_signed=True)
                        qn.params['weight_block_size'] = w.block_size
                    else:
                        w.scale, w.zerop, w.qmin, w.qmax, w.dtype = get_linear_quant_params_from_tensor(
                            w, q_mode_weight, w.qbits, is_signed=True)
                    w.betensor = linear_quantize_clip(w.betensor, w.broadcast_scale, w.broadcast_zerop, w.qmin, w.qmax)
                    # currently lib can detect weight only quantization through IR Dtype info
                    # qn.params['approximate_method'] = 'weight_only_quantization'
                if qn.get_param('is_perf_mode', optional=True, default_value=True):
                    qn.approximate()

        for nkey, nvalue in new_cast_edges_in_qg.items():
            edge, consumer_node, edge_s, consumer_node_s = nkey
            nattrs = nvalue[0]
            from_dtype = nvalue[1]
            to_dtype = nvalue[-1]

            def insert_cast_edge_to_graph(e, e_s, cn, con_node):
                cn.add_input(e)
                cn.add_output(e_s)
                cn.params['to_dtype'] = e.dtype
                cn.params['ignore_scale_zp'] = True
                cn.params['clip_mode'] = 'TRUNCATION' if not is_float(e.dtype) else 'SATURATION'
                e_idx = con_node.remove_input(e)
                con_node.add_input(e_s, e_idx)
            dummy_op = PyNode(self.get_valid_node_name(edge.pnode.name), OpType.Cast)
            dummy_op.additional = True
            dummy_op.attrs.update(nattrs.clone())
            atensor_name = self.get_valid_tensor_name(edge.name)
            atensor = edge.clone(atensor_name)
            if self.quantgraph != self:
                # maintain one-to-one correspondence
                cn_s = dummy_op.clone(dummy_op.name)
                at_s = atensor.clone(atensor.name)
                insert_cast_edge_to_graph(edge_s, at_s, cn_s, consumer_node_s)
                self.add_node(cn_s)
            insert_cast_edge_to_graph(edge, atensor, dummy_op, consumer_node)
            edge.dtype = from_dtype
            dummy_op.params['to_dtype'] = to_dtype
            atensor.dtype = to_dtype
            dummy_op.params['unquantifiable'] = True
            self.quantgraph.add_node(dummy_op)

    def qforward(self, feed_data, disable_pbar=True, keep_tensors=False):
        return self.quantgraph.forward(feed_data, disable_pbar, keep_tensors)

    def graph_param_show(self, *args):
        from AIPUBuilder.Optimizer.logger import OPT_DEBUG
        if self.quantgraph is not None:
            for n in self.quantgraph.nodes:
                if n.quantized:
                    n.param_show(*args)
                    OPT_DEBUG('\b'*36+'\n')
        else:
            OPT_DEBUG('no quantgraph when graph params show, please check workflow')

    def insert_dummy_node_ahead(self, ntype, condition_func=lambda node, parent_node, edge_tensor: False):
        from AIPUBuilder.Optimizer.utils.string_utils import timestamp_string
        from AIPUBuilder.Optimizer.framework.pycore.pynode import PyNode
        cast_count_num = 0
        inserted_op_list = []
        for n in self.nodes:
            for inp_t in n.inputs:
                parent_node = None
                if inp_t.pnode:
                    parent_node = inp_t.pnode
                else:
                    for parent in n.parents:
                        if inp_t in parent.outputs:
                            parent_node = parent
                if parent_node and condition_func(n, parent_node, inp_t):
                    _nname = parent_node.name + ("_%s_" % (str(ntype)[7:],)) + str(cast_count_num) + timestamp_string()
                    index = parent_node.outputs.index(inp_t)
                    dummy_op = PyNode(self.get_valid_node_name(_nname), ntype)
                    dummy_op.additional = True
                    dummy_op.add_input(inp_t)
                    atensor_name = self.get_valid_tensor_name(inp_t.name + ("_%s_tensor_" % (str(ntype)[7:],)) +
                                                              str(cast_count_num) + timestamp_string())
                    atensor = inp_t.clone(atensor_name)
                    dummy_op.add_output(atensor)
                    idx = n.remove_input(inp_t)
                    n.add_input(atensor, idx)
                    # because the output tensor of dummy_op is clone from the output of parent_node, so using
                    # the attributes of parent_node to quantize is reasonable
                    dummy_op.attrs.update(parent_node.attrs.clone())
                    inserted_op_list.append(dummy_op)
                    cast_count_num += 1
                    self.add_node(dummy_op)
        return inserted_op_list

    @staticmethod
    def deduce_quantization_infos(graph):
        from AIPUBuilder.Optimizer.framework import PyNode
        from AIPUBuilder.Optimizer.logger import OPT_ERROR, OPT_WARN
        import torch

        if graph is None:
            OPT_ERROR(f"please check the graph(==None) before deduce quantization information.")
            return None

        for node in graph.nodes:
            PyNode.deduce_quantization_infos(node)

        for subgraph in graph.subgraph_map.values():
            for node in subgraph.nodes:
                PyNode.deduce_quantization_infos(node)
