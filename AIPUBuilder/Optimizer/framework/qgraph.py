# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    with tqdm(dataloader, desc=desc, file=sys.stdout, disable=disable_tqdm) as pbar:
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
        feed_data = inputs_data
        if opt_use_cuda():
            if isinstance(feed_data, dict):
                feed_data = {key: feed_data[key].cuda() for key in feed_data}
            elif isinstance(feed_data, (list, tuple)):
                feed_data = [ii.cuda() for ii in feed_data]
            else:
                feed_data = feed_data.cuda()
        data = feed_data
        if len(self.input_tensors) == 1 and not isinstance(feed_data, list):
            data = [feed_data, ]
        for inp, d in zip(self.input_tensors, data):
            inp.betensor = d
            inp.betensor = inp.betensor.float()

    def statistic(self, config):
        import sys
        from AIPUBuilder.Optimizer.framework.pycore.pytensor import TensorShape
        from AIPUBuilder.Optimizer.utils import QuantMode
        from AIPUBuilder.Optimizer.config import CalibrationStrategyField, TrimInfinityField
        from AIPUBuilder.Optimizer.logger import tqdm, OPT_DEBUG

        time_saving_mode = not config.save_statistic_info
        trim_inf_c, trim_inf_v = TrimInfinityField.parse(config.trim_infinity_before_statistic)
        trim_inf = {}
        for n in self.nodes:
            dv = ((float('-inf'), float('inf')), '')
            if trim_inf_c == 1:
                trim_inf[n] = trim_inf_v
            elif trim_inf_c == 2:
                tname = n.type.name.lower().strip()
                trim_inf[n] = trim_inf_v[tname] if tname in trim_inf_v.keys() else dv
            elif trim_inf_c == 3:
                layer_id = int(n.attrs['layer_id'])
                trim_inf[n] = trim_inf_v[layer_id] if layer_id in trim_inf_v.keys() else dv
            else:
                trim_inf[n] = dv
            if trim_inf[n] != dv:
                OPT_DEBUG(
                    f"{n.type}, layer_id={n.attrs['layer_id']}, layer_name={n.name}, infinite values will be trimed {trim_inf[n]} before statistic", log_once=True)

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
                            key_axis_c = v.key_axis_c if v.ir_shape != TensorShape([]) else None
                            if time_saving_mode:
                                if not (QuantMode.is_per_channel(qmethod_wht) or n.get_param('group', optional=True, default_value=1) > 1):
                                    key_axis_c = None
                                r = CalibrationStrategyField._need_statistic_info(cstrategy)
                                if not r['histc']:
                                    histc_bins = None
                                if not r['std_mean']:
                                    statistic_std_mean = False
                            v.statistic(running_statistic_momentum, key_axis=key_axis_c,
                                        histc_bins=histc_bins, statistic_std_mean=statistic_std_mean,
                                        trim_infinity=trim_inf[n],
                                        reset=True)
                    pbar.update(1)
                pbar.refresh()
        self.constants_statisticed = True

        for n in self.nodes:
            running_statistic_momentum = n.attrs["running_statistic_momentum"]
            histc_bins = n.attrs["histc_bins"]
            statistic_std_mean = True
            astrategy = n.get_attrs('q_strategy_activation')
            qmethod_act = n.get_attrs('q_mode_activation')
            if not n.quantized:
                r = CalibrationStrategyField._need_statistic_info(astrategy)
                if time_saving_mode:
                    if not r['histc']:
                        histc_bins = None
                    if not r['std_mean']:
                        statistic_std_mean = False
                for o in n.outputs:
                    o.statistic(running_statistic_momentum, key_axis=None,
                                histc_bins=histc_bins, statistic_std_mean=statistic_std_mean,
                                trim_infinity=trim_inf[n],
                                reset=not self.current_batch_idx)
                for p in n.placeholders:
                    p.statistic(running_statistic_momentum, key_axis=None,
                                histc_bins=histc_bins, statistic_std_mean=statistic_std_mean,
                                trim_infinity=trim_inf[n],
                                reset=not self.current_batch_idx)

    def save_statistic_info(self, statistic_info_fname):
        import pickle
        from AIPUBuilder.Optimizer.logger import OPT_WARN
        from AIPUBuilder.Optimizer.utils.files_utils import make_path

        def get_value(t_param):
            return t_param.cpu().contiguous().numpy() if t_param is not None else t_param

        statistic_info = {}
        for n in self.nodes:
            statistic_info[n.name] = {}
            for o in n.outputs:
                statistic_info[n.name][o.name] = {
                    "extrema_min": o.extrema_min,
                    "extrema_max": o.extrema_max,
                    "running_min": o.running_min,
                    "running_max": o.running_max,
                    "running_mean": o.running_mean,
                    "running_std": o.running_std,
                    "running_mad": o.running_mad,
                    "running_histc": get_value(o.running_histc),
                }
            for _, v in n.constants.items():
                statistic_info[n.name][v.name] = {
                    "extrema_min": v.extrema_min,
                    "extrema_max": v.extrema_max,
                    "running_min": v.running_min,
                    "running_max": v.running_max,
                    "running_mean": v.running_mean,
                    "running_std": v.running_std,
                    "running_mad": v.running_mad,
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
                    "extrema_min": p.extrema_min,
                    "extrema_max": p.extrema_max,
                    "running_min": p.running_min,
                    "running_max": p.running_max,
                    "running_mean": p.running_mean,
                    "running_std": p.running_std,
                    "running_mad": p.running_mad,
                    "running_histc": get_value(p.running_histc),
                }
        statistic_info_fname = make_path(statistic_info_fname)
        try:
            # np.save(statistic_info_fname, statistic_info)
            with open(statistic_info_fname, 'wb') as fw:
                pickle.dump(statistic_info, fw)
        except Exception as e:
            OPT_WARN(f"Optimizer saves the statistic file failed, because {e}")

    def load_statistic_info(self, statistic_info_fname):
        import numpy as np
        import torch
        from AIPUBuilder.Optimizer.logger import OPT_FATAL, OPT_INFO
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
        for n in self.nodes:
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
        for n in self.nodes:
            qn = n.clone(n.name+"_clone_")
            qn.quantize()
            for i, t in enumerate(n.outputs):
                tc = qn.outputs[i]
                t.scale = tc.scale
                t.zerop = tc.zerop
                t.qbits = tc.qbits
                t.dtype = tc.dtype
                t.qmin = tc.qmin
                t.qmax = tc.qmax
                t.qinvariant = tc.qinvariant
            for i, t in enumerate(n.placeholders):
                tc = qn.placeholders[i]
                t.scale = tc.scale
                t.zerop = tc.zerop
                t.qbits = tc.qbits
                t.dtype = tc.dtype
                t.qmin = tc.qmin
                t.qmax = tc.qmax
                t.qinvariant = tc.qinvariant
            for k, t in n.constants.items():
                if k in qn.constants.keys():
                    tc = qn.constants[k]
                    t.scale = tc.scale
                    t.zerop = tc.zerop
                    t.qbits = tc.qbits
                    t.dtype = tc.dtype
                    t.qmin = tc.qmin
                    t.qmax = tc.qmax
                    t.qinvariant = tc.qinvariant

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
        from AIPUBuilder.Optimizer.logger import tqdm
        if self.quantgraph is None:
            self.quantgraph = self.clone()

        with tqdm(total=len(self.quantgraph.nodes), desc='quantize each layer', file=sys.stdout, leave=True) as pbar:
            for n in self.quantgraph.nodes:
                if not n.quantized:
                    n.quantize()
                    n.quantized = True
                pbar.update(1)
            pbar.refresh()

    def qforward(self, feed_data, disable_pbar=True):
        return self.quantgraph.forward(feed_data, disable_pbar)

    def graph_param_show(self, *args):
        from AIPUBuilder.Optimizer.logger import OPT_DEBUG
        if self.quantgraph is not None:
            for n in self.quantgraph.nodes:
                if n.quantized:
                    n.param_show(*args)
                    OPT_DEBUG('\b'*36+'\n')
        else:
            OPT_DEBUG('no quantgraph when graph params show, please check workflow')

    def insert_pad_op_ahead(self, condition_func=lambda node, parent_node, edge_tensor: False):  # for avgpool cnt=ceil=true
        from AIPUBuilder.Optimizer.framework.pycore.pytype import OpType
        inserted_op_list = self.insert_dummy_node_ahead(OpType.Pad, condition_func)
        for n in inserted_op_list:
            n.params['mode'] = 'CONSTANT'
            n.params['constant_value'] = 0 - n.children[0].inputs[0].zerop
            pool_params = n.children[0].params
            if n.children[0].type == OpType.Pooling:
                n.params['pads'] = [[0, 0], [pool_params['pad_top'], pool_params['pad_bottom']],
                                    [pool_params['pad_left'], pool_params['pad_right']], [0, 0]]
                n.children[0].params['pad_left'] = 0
                n.children[0].params['pad_right'] = 0
                n.children[0].params['pad_top'] = 0
                n.children[0].params['pad_bottom'] = 0
            if n.children[0].type == OpType.Pooling3D:  # NDHWC
                n.params['pads'] = [[0, 0], [pool_params['pad_z_begin'], pool_params['pad_z_end']],
                                    [pool_params['pad_y_begin'], pool_params['pad_y_end']],
                                    [pool_params['pad_x_begin'], pool_params['pad_x_end']], [0, 0]]
                n.children[0].params['pad_x_begin'] = 0
                n.children[0].params['pad_x_end'] = 0
                n.children[0].params['pad_y_begin'] = 0
                n.children[0].params['pad_y_end'] = 0
                n.children[0].params['pad_z_begin'] = 0
                n.children[0].params['pad_z_end'] = 0
        return inserted_op_list

    # A->B => A->cast->B
    def insert_cast_op_ahead(self, condition_func=lambda node, parent_node, edge_tensor: False):
        from AIPUBuilder.Optimizer.framework.pycore.pytype import OpType
        from AIPUBuilder.Optimizer.utils.dtype_utils import bits2dtype, is_signed
        inserted_op_list = self.insert_dummy_node_ahead(OpType.Cast, condition_func)
        for n in inserted_op_list:
            # set cast default value
            n.params['only_for_quantized'] = True
            n.params['to_dtype'] = bits2dtype(n.attrs['q_bits_activation'], is_signed=is_signed(n.outputs[0].dtype))
        return inserted_op_list

    def insert_dummy_node_ahead(self, ntype, condition_func=lambda node, parent_node, edge_tensor: False):
        from AIPUBuilder.Optimizer.utils.string_utils import timestamp_string
        from AIPUBuilder.Optimizer.framework.pycore.pynode import PyNode
        import copy
        node_idx = 0
        cast_count_num = 0
        end_idx = len(self.nodes)
        inserted_op_list = []
        if condition_func:
            while node_idx < end_idx:
                n = self.nodes[node_idx]
                ni = len(n.inputs)
                for tensor_idx in range(ni):
                    current_parent = None
                    for parent in n.parents:
                        if n.inputs[tensor_idx] in parent.outputs:
                            current_parent = parent
                            # p_idx = parent.outputs.index(n.inputs[tensor_idx])
                            break

                    if current_parent is not None and condition_func(n, current_parent, n.inputs[tensor_idx]):
                        parent_out_tensor = n.inputs[tensor_idx]
                        _nname = current_parent.name + \
                            ("_%s_" % (str(ntype)[7:],)) + str(cast_count_num) + timestamp_string()
                        dummy_op = PyNode(self.get_valid_node_name(_nname), ntype)
                        dummy_op.additional = True
                        dummy_op.add_input(parent_out_tensor)
                        atensor_name = self.get_valid_tensor_name(
                            parent_out_tensor.name + ("_%s_tensor_" % (str(ntype)[7:],)) + str(cast_count_num) + timestamp_string())
                        atensor = parent_out_tensor.clone(atensor_name)
                        dummy_op.add_output(atensor)
                        idx = n.remove_input(parent_out_tensor)
                        n.add_input(atensor, idx)
                        dummy_op.attrs.update(n.attrs.clone())
                        if 'quantization_info' in n.attrs:
                            dummy_op.attrs['quantization_info'] = {}
                            dummy_op.attrs['quantization_info'][atensor_name] = current_parent.attrs['quantization_info'][parent_out_tensor.name]
                        dummy_op.attrs['layer_id'] = '0' + str(current_parent.attrs['layer_id'])
                        self.nodes.insert(node_idx, dummy_op)
                        # update graph network relations and related variables
                        self.init_networkx()
                        inserted_op_list.append(dummy_op)
                        node_idx += 1
                        end_idx += 1
                        cast_count_num += 1
                node_idx += 1
        return inserted_op_list
