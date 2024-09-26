# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

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
                        for _, v in n.constants.items():
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

    def set_tensor_quantization_attrs(self):
        import sys
        from AIPUBuilder.Optimizer.logger import tqdm
        with tqdm(total=len(self.nodes), desc='update_tensor_quantization_attrs', file=sys.stdout, leave=False) as pbar:
            for n in self.nodes:
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
                t.scale = 1.0
                t.zerop = 0
                t.qbits = None
                t.dtype = t.ir_dtype
                t.qmin = None
                t.qmax = None
                t.qinvariant = None

    def quantize(self):
        import sys
        from AIPUBuilder.Optimizer.logger import tqdm, OPT_WARN, OPT_DEBUG
        from AIPUBuilder.Optimizer.framework import OpType, Dtype, PyTensor
        from AIPUBuilder.Optimizer.utils import (is_float, str2dtype, QuantMode,
                                                 get_linear_quant_params_from_tensor,
                                                 torch_type2dtype,
                                                 linear_quantize_clip)

        if self.quantgraph is None:
            self.quantgraph = self.clone()

        self.quantgraph.set_tensor_quantization_attrs()  # pylint: disable=no-member

        # record the map between quantized node's name and source node object pointer
        qnmap = {}
        for qn, n in zip(self.quantgraph.nodes, self.nodes):
            qnmap[qn.name] = n
        for qn in self.quantgraph.nodes:
            qn.attrs['map_to_original_node'] = qnmap

        with tqdm(total=len(self.quantgraph.nodes), desc='quantize each layer', file=sys.stdout, leave=True) as pbar:
            for n in self.quantgraph.nodes:
                if not n.quantized:
                    n.quantize()
                    # n.quantized = True
                pbar.update(1)
            pbar.refresh()

        graph = self.quantgraph
        for qn in graph.nodes:
            fd = qn.attrs['trigger_float_op'].name if isinstance(qn.attrs['trigger_float_op'], Dtype) \
                else str(qn.attrs['trigger_float_op']).lower().strip()
            if fd != 'disable' and qn.params['unquantifiable']:
                if qn.type == OpType.Quantize:
                    qn.params['quantize_scale'] = qn.outputs[0].scale
                    qn.params['quantize_zp'] = qn.outputs[0].zerop
                if qn.type == OpType.DeQuantize:
                    qn.params['quantize_scale'] = qn.inputs[0].scale
                    qn.params['quantize_zp'] = qn.inputs[0].zerop

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
                        t.scale = 1.0
                        t.zerop = 0
                        t.qbits = None
                        t.dtype = t.ir_dtype
                        t.qmin = None
                        t.qmax = None
                        t.qinvariant = None

                o_dtype = str2dtype(fd)
                for ot in qn.outputs:
                    if is_float(ot.dtype):
                        ot.dtype = o_dtype
                if qn.type == OpType.Cast:
                    qn.params['to_dtype'] = qn.outputs[0].dtype

                if qn.type != OpType.Quantize and qn.type != OpType.DeQuantize:  # per-channel scale/zp would be in constants
                    for key, ct in qn.constants.items():
                        if is_float(ct.ir_dtype):
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
                qn.approximate()

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
                if is_float(t.dtype):
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

        if graph is None:
            OPT_ERROR(f"please check the graph(==None) before deduce quantization information.")
            return None

        for node in graph.nodes:
            node.quantized = True
            if node.type in [OpType.Quantize]:
                _deduce_quantization_info_to_tensor_from_ir(node, get_tensor_default_property())
                continue

            if node.get_param('unquantifiable', optional=True, default_value=False):
                node.quantized = False
                if node.get_param('is_perf_mode', optional=True, default_value=False):
                    node.approximated = True
                if 'weights' in node.constants.keys():
                    wt = node.constants["weights"]
                    if not is_float(wt.dtype):
                        node.attrs['weight_only_quantization'] = True
            else:
                dtypes = [t.dtype for t in (list(node.outputs) + list(node.inputs))]
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
                        if is_float(dt):
                            node.quantized = False
                            break
            if node.quantized:
                _deduce_quantization_info_to_tensor_from_ir(node, get_tensor_default_property())
