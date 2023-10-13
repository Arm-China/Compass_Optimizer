# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

import torch
import numpy as np
import os
import sys
import copy
import json
import functools
from torch.utils.data import DataLoader
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.config import *
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.ops import *
from AIPUBuilder.Optimizer.analyzer import *
from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.features import *
from AIPUBuilder.Optimizer.passes import *


class OptMaster(object):

    def __init__(self, graph, hparams=None, calibration_dataloader=None, validation_dataloader=None,
                 validation_metrics=[]):
        self.g = graph
        self.calibration_dataloader = calibration_dataloader
        self.validation_dataloader = validation_dataloader
        self.hparams = hparams
        self.validation_metrics = validation_metrics
        self.f_metrics = []
        self.q_metrics = []
        # ----------------------------------------------------------------------------
        # |         | add/delete node | add/delete node | change quantization params |
        # |         | of float graph  | of quant graph  | (min/max, scales, etc)     |
        # ----------------------------------------------------------------------------
        # prepare datasets, metrics and set configurations
        # ----------------------------------------------------------------------------
        # | Stage 1 | allowed         | forbidden       | forbidden                  |
        # pre-quantization optimization
        # ----------------------------------------------------------------------------
        # statistic & initial calibration
        # ----------------------------------------------------------------------------
        # | Stage 2 | allowed         | forbidden       | allowed                    |
        # quantization aware optimization
        # ----------------------------------------------------------------------------
        # final calibration & quantize
        # ----------------------------------------------------------------------------
        # | Stage 3 |forbidden        | allowed         | forbidden                  |
        # post-quantization optimization
        # ----------------------------------------------------------------------------
        self.graph_optimize_stage1_flag = False
        self.graph_optimize_stage2_flag = False
        self.graph_optimize_stage3_flag = False
        # if 'ts_min_file' in hparams.__dict__ and 'ts_max_file' in hparams.__dict__ :
        #     self.ts_min_file = hparams.ts_min_file
        #     self.ts_max_file = hparams.ts_max_file
        # else:
        #     self.ts_min_file = ""
        #     self.ts_max_file = ""
        self.dataloader4debug = None
        self.calibration_dataloader = None
        self.validation_dataloader = None
        self.fake_quant_scopes = []
        self.op_need_cast_dtypes_for_lib = set()
        self.batch_size_in_IR = 1

    def prepare(self, argv):
        """prepare the calibration and validation dataset, the metric method, and config.json
        :param argv: config in cfg file
        :return:
        """
        config_info = get_info_from_graph(self.g, batch_dim=argv.data_batch_dim)
        if argv.without_batch_dim:
            config_info['batch_size'] = 0
            argv.calibration_batch_size = 1
            argv.metric_batch_size = 1
        self.batch_size_in_IR = config_info['batch_size']
        # this only uses to check beteween the return item len and the input Op num.
        checker_dataloader = None
        collate_fn = None
        if argv.dataset != '' and argv.calibration_data != '':
            if argv.dataset.lower() in QUANTIZE_DATASET_DICT:
                QUANTIZE_DATASET_DICT[argv.dataset.lower()].data_batch_dim = argv.data_batch_dim
                QUANTIZE_DATASET_DICT[argv.dataset.lower()].label_batch_dim = argv.label_batch_dim
                caldataset = QUANTIZE_DATASET_DICT[argv.dataset.lower()](argv.calibration_data)
                collate_fn = caldataset.collate_fn if hasattr(caldataset, 'collate_fn') else None
                self.calibration_dataloader = DataLoader(caldataset,
                                                         batch_size=argv.calibration_batch_size,
                                                         shuffle=argv.calibration_shuffe,
                                                         num_workers=argv.dataloader_workers,
                                                         collate_fn=collate_fn)
                self.dataloader4debug = DataLoader(caldataset,
                                                   batch_size=min(len(caldataset), max(1, self.batch_size_in_IR)),
                                                   shuffle=False,
                                                   num_workers=argv.dataloader_workers,
                                                   collate_fn=collate_fn)

                checker_dataloader = DataLoader(caldataset,
                                                batch_size=min(len(caldataset), max(1, self.batch_size_in_IR)),
                                                shuffle=False,
                                                num_workers=argv.dataloader_workers,
                                                collate_fn=collate_fn)

        # if metric, opt will create [validation_dataset, metrics]
        if argv.metric != '':
            valdataset = QUANTIZE_DATASET_DICT[argv.dataset.lower()](argv.data, argv.label)
            self.validation_dataloader = DataLoader(valdataset,
                                                    batch_size=argv.metric_batch_size,
                                                    shuffle=False,
                                                    num_workers=argv.dataloader_workers,
                                                    collate_fn=collate_fn)
            self.dataloader4debug = DataLoader(valdataset,
                                               batch_size=min(len(valdataset), max(1, self.batch_size_in_IR)),
                                               shuffle=False,
                                               num_workers=argv.dataloader_workers,
                                               collate_fn=collate_fn)
            checker_dataloader = DataLoader(valdataset,
                                            batch_size=min(len(valdataset), max(1, self.batch_size_in_IR)),
                                            shuffle=False,
                                            num_workers=argv.dataloader_workers,
                                            collate_fn=collate_fn)

            # get metric instance
            from AIPUBuilder.Optimizer.config.cfg_fields import MetricField
            m = argv.metric.replace(' ', '')
            metrics = MetricField._split_metrics(m)
            func_args = MetricField._get_func_args(metrics)

            # delete the repeat metric
            fn_arg_dict = {}
            fas = [[s[0].lower(), s[1]] for s in func_args]
            repeat = False
            for fa in fas:
                fname = fa[0]
                args = fa[1]
                if fname in fn_arg_dict.keys():
                    for argl in fn_arg_dict[fname]:
                        if args == argl:
                            repeat = True
                            break
                    if not repeat:
                        fn_arg_dict[fname].append(args)
                else:
                    fn_arg_dict.update({fname: [args]})

            for fn, argl in fn_arg_dict.items():
                for arg in argl:
                    self.validation_metrics.append(QUANTIZE_METRIC_DICT[fn.lower()](*arg))

            for metric in self.validation_metrics:
                metric.reset()
                self.f_metrics.append(copy.deepcopy(metric))
                metric.reset()
                self.q_metrics.append(copy.deepcopy(metric))

        # check dataset has the same data items with the input tensors.
        if checker_dataloader != None:
            inp_tensor_num = len(self.g.input_tensors)
            dataset_sample_len = 0
            for i, sample in enumerate(checker_dataloader):
                dataset_sample_len = 1
                if isinstance(sample[0], list):
                    dataset_sample_len = len(sample[0])
                break
            if inp_tensor_num != dataset_sample_len:
                OPT_ERROR('Dataset plugin returns the inputs num(=%d) != len(input_tensors)(=%d)' %
                          (dataset_sample_len, inp_tensor_num))

        if self.hparams.__getattribute__('opt_config') != '':
            opt_config_file = self.hparams.opt_config
            # TODO
            try:
                with open(opt_config_file, 'r') as fr:
                    info = json.load(fr)
                    # TODO
                    info = filter_valid_properties(info, initial_info=config_info)
                    for lname, properties in info.items():
                        if lname not in config_info:
                            config_info[lname] = {}
                        for k, v in properties.items():
                            config_info[lname].update({k: v})
            except:
                OPT_FATAL("Invalid opt_config file! please refer to opt_template.json")
        # process trigger float op attr
        from AIPUBuilder.Optimizer.config.cfg_fields import TriggerFloatOpField as tfof
        tf_name, tf_ops, tf_alt = tfof._parse(self.hparams.trigger_float_op)[1][0]
        for node in self.g.nodes:
            properties = config_info[node.name]

            def init_attrs(key, default_value=None):
                node.attrs[key] = properties[key] if key in properties else default_value

            init_attrs('layer_id')
            init_attrs('layer_top_type_original')
            init_attrs('q_mode_activation', self.hparams.quantize_method_for_activation)
            init_attrs('q_mode_weight', self.hparams.quantize_method_for_weight)
            if node.type in [OpType.DepthwiseConv] and 'q_mode_weight' not in properties:
                node.attrs['q_mode_weight'] = 'per_channel_symmetric_restricted_range'
            # bias and weight better to have the same quantize_method
            init_attrs('q_mode_bias', node.attrs['q_mode_weight'])
            init_attrs('q_bits_activation', self.hparams.activation_bits)
            init_attrs('q_bits_weight', self.hparams.weight_bits)
            if node.type == OpType.BatchNorm and 'q_bits_weight' not in properties:
                node.attrs['q_bits_weight'] = 16
            init_attrs('q_bits_bias', self.hparams.bias_bits)
            init_attrs('q_strategy_activation', self.hparams.calibration_strategy_for_activation)
            init_attrs('q_strategy_weight', self.hparams.calibration_strategy_for_weight)
            init_attrs('q_strategy_bias', self.hparams.calibration_strategy_for_weight)
            init_attrs('running_statistic_momentum', self.hparams.running_statistic_momentum)
            init_attrs('histc_bins', self.hparams.histc_bins)
            init_attrs('debug_fake_quantize', False)
            if node.type in [OpType.Resize, OpType.Interp, ]:
                init_attrs('resize_degrade_to_nearest', self.hparams.resize_degrade_to_nearest)
            if node.type in [OpType.Convolution, OpType.DepthwiseConv, OpType.ConvTranspose, OpType.Convolution3D,
                             OpType.ConvTranspose3D]:
                init_attrs('with_winograd', self.hparams.with_winograd),
            init_attrs('lut_items_in_bits', self.hparams.lut_items_in_bits)
            init_attrs('multiplier_bits', node.attrs['q_bits_activation'] if self.hparams.multiplier_bits == ''
                       else self.hparams.multiplier_bits)
            init_attrs('force_dtype_int', self.hparams.force_dtype_int)
            init_attrs('force_shift_positive', self.hparams.force_shift_positive)
            init_attrs('min_compatible_zhouyi_target', self.hparams.min_compatible_zhouyi_target)
            # init_attrs('force_dtype_int', False)
            init_attrs('bias_effective_bits', node.attrs['q_bits_bias'] if self.hparams.bias_effective_bits == ''
                       else self.hparams.bias_effective_bits)
            init_attrs('unify_shifts_for_aiff', self.hparams.unify_shifts_for_aiff)
            init_attrs('trigger_float_op', tf_name)
            if len(tf_ops) > 0 and not (node.type.name.lower() in tf_ops or int(node.attrs['layer_id']) in tf_ops):
                init_attrs('trigger_float_op', tf_alt)
            if node.type == OpType.Concat:
                init_attrs('unify_scales_for_multi_inputs_operator_threshold',
                           self.hparams.unify_scales_for_concat_threshold)
            node.attrs['optimization_info'] = {}
            node.attrs['batch_size_in_IR'] = self.batch_size_in_IR
            node.attrs['calculate_running_time'] = False

            # update the scaling_bits to corresponding ops
            # from AIPUBuilder.Optimizer.cfg_parser import ScalingBits
            file_configured_scaling_bits = properties['scaling_bits'] if 'scaling_bits' in properties else None
            if file_configured_scaling_bits is None:
                ScalingBitsField._update_to_node_attr(node, self.hparams.scaling_bits)
            else:
                node.attrs['scaling_bits'] = properties['scaling_bits']

            # check if each op exists in registered dict
            from AIPUBuilder.Optimizer.framework import OP_DICT, QUANT_OP_DICT
            if node.type not in OP_DICT:
                OPT_ERROR('unsupported op "%s", can not find it in OP_DICT, please implement this op firstly' % str(
                    node.type))
            if node.type not in QUANT_OP_DICT:
                OPT_ERROR(
                    'unsupported op "%s", can not find it in QUANT_OP_DICT, please implement this op firstly' % str(
                        node.type))

        # do a forward to check graph firstly and init placeholders,
        # force each op to be able to handle all zero inputs.
        inputs = []
        if self.hparams.run_mode != 'quant_ir_forward' and \
           ((hasattr(self.g, 'compat_quantized_model') and not self.g.compat_quantized_model) or not hasattr(self.g, 'compat_quantized_model')):
            for inp in self.g.input_tensors:
                shape = list(inp.ir_shape)
                dtype = inp.dtype
                inputs.append(np.zeros(shape).astype(dtype2nptype(dtype)))
            OPT_INFO("init graph by forwarding one sample filled with zeros")
            self.g.current_batch_size = self.batch_size_in_IR
            self.g.current_batch_idx = 0
            self.g.forward(inputs, disable_pbar=False)

        # parse fake_quant_scopes params
        self.fake_quant_scopes = []
        scopes_str_list = re.findall(r'\d+', argv.fake_quant_scopes_for_debug)
        for k in range(0, len(scopes_str_list), 2):
            start = int(scopes_str_list[k])
            end = int(scopes_str_list[k + 1])
            self.fake_quant_scopes.append((start, end))
        scopes_op_list = [x.lower().strip() for x in re.split(
            r',|\s+', argv.fake_quant_operators_for_debug) if x.lower().strip()]
        for node in self.g.nodes:
            if str(node.type)[7:].lower() in scopes_op_list:
                start = int(node.attrs['layer_id'])
                self.fake_quant_scopes.append((start, start))

        # assign start nodes (input layers or constant layers) that should be qinvariant
        sn_str_list = re.findall(r'\d+', str(argv.set_qinvariant_for_start_nodes))
        sn_list = [x.strip() for x in sn_str_list]
        for node in self.g.nodes:
            if node.type in [OpType.Input, OpType.Constant] and str(node.attrs['layer_id']).strip() in sn_list:
                top_type_original = node.attrs['layer_top_type_original'][0]
                original_top_dtype = str2dtype(top_type_original)
                is_original_top_float = is_float(original_top_dtype)
                if is_original_top_float:
                    node.attrs['layer_top_type_original'][0] = dtype2str(Dtype.INT32)

    def optimize(self):
        self.graph_optimize_stage1()
        self.statistic()
        self.graph_optimize_stage2()
        self.quantize()
        self.collect_information_for_debug()
        self.graph_optimize_stage3()

        self.graph_param_show()

    def graph_param_show(self, *args):
        self.g.graph_param_show(*args)

    @opt_workflow_register
    def graph_optimize_stage1(self):
        # hardware independent optimization
        if self.graph_optimize_stage1_flag:
            return

        optimization_stage1(self.g, self.hparams)
        # must call self.g.topological_sort() if nodes were added or deleted here.

        self.graph_optimize_stage1_flag = True

    @opt_workflow_register
    def graph_optimize_stage2(self):
        # quantization aware optimization (also partly hardware aware)
        if self.graph_optimize_stage2_flag:
            return
        if self.hparams.save_statistic_info:
            # save statistic info first
            self.g.save_statistic_info(self.hparams.output_dir + "/" + self.hparams.model_name + "_statistic_info.npy")
        # apply calibration strategy per-layer
        if self.hparams.calibration_strategy_for_weight and self.hparams.calibration_strategy_for_activation:
            OPT_INFO('applying calibration strategy based on statistic info')
            with tqdm(total=len(self.g.nodes), desc='calibration', file=sys.stdout) as pbar:
                for n in self.g.nodes:
                    astrategy = n.attrs[
                        'q_strategy_activation'] if 'q_strategy_activation' in n.attrs else self.hparams.calibration_strategy_for_activation
                    cstrategy = n.attrs[
                        'q_strategy_weight'] if 'q_strategy_weight' in n.attrs else self.hparams.calibration_strategy_for_weight
                    qmethod_wht = n.attrs[
                        'q_mode_weight'] if 'q_mode_weight' in n.attrs else self.hparams.quantize_method_for_weight
                    qmethod_act = n.attrs[
                        'q_mode_activation'] if 'q_mode_activation' in n.attrs else self.hparams.quantize_method_for_activaion
                    for o in n.outputs:
                        o.qbits = n.attrs['q_bits_activation']
                        apply_calibration_strategy(o, astrategy, qmethod_act)
                    for k, v in n.constants.items():
                        v.qbits = n.attrs['q_bits_bias'] if k.lower() == 'biases' else n.attrs['q_bits_weight']
                        apply_calibration_strategy(v, cstrategy, qmethod_wht)
                    for p in n.placeholders:
                        p.qbits = n.attrs['q_bits_activation']
                        apply_calibration_strategy(p, astrategy, qmethod_act)
                    pbar.update(1)
                pbar.refresh()

        if self.hparams.global_calibration.lower().strip() != 'none':
            # get the intial calibration's results (each tensor's scale, zp, dtype, qbits)
            self.g.set_tensor_quantization_attrs()
            # apply global quantization optimization (scales, rounding, etc) here
            apply_global_calibration(self.g, self.calibration_dataloader, self.hparams.global_calibration)
            # clear float graph's calibration results (each tensor's scale, zp, dtype, qbits) to avoid misusing in float forward
            self.g.clear_tensor_quantization_attrs()

        optimization_stage2(self.g, self.hparams)

        self.graph_optimize_stage2_flag = True

    @opt_workflow_register
    def graph_optimize_stage3(self):
        # hardware aware optimization (quantization independent)
        if self.graph_optimize_stage3_flag:
            return

        ################################################################
        #  insert pad op for avgpool when count_include_pad=ceil_mode=True zp!=0
        self.g.quantgraph.insert_pad_op_ahead(
            condition_func=lambda node, parent_node, edge_tensor: node.type in OP_NEED_ADD_PAD_AVOID_ASYNC_DIVIDE and
            node.get_param('ceil_mode', optional=True, default_value=False) == True and
            node.get_param('count_include_pad', optional=True, default_value=False) == True and
            node.get_param('method') == 'AVG' and node.outputs[0].zerop != 0)

        ################################################################
        # transform useless op to lightweight reshape op
        need_replace_ops = []
        for n in self.g.quantgraph.nodes:
            if n != None:
                if (n.type == OpType.Cast and n.parents[0].attrs['q_mode_activation'] == n.attrs[
                    'q_mode_activation'] and len(n.inputs) > 0 and len(n.outputs) > 0 and n.inputs[0].dtype ==
                    n.outputs[0].dtype) \
                        or (n.type == OpType.FakeQuantWithMinMaxVars):
                    # create reshape node
                    transform_op = PyNode(n.name, OpType.Reshape)
                    transform_op.additional = True
                    # set attrs and params
                    transform_op.attrs.update(n.attrs.clone())
                    transform_op.params['shape'] = n.outputs[0].ir_shape
                    # record pairs
                    need_replace_ops.append((n, transform_op))
        for old, new in need_replace_ops:
            self.g.quantgraph.replace_node_safely(old, new)

        ##############seperate featuremap ############################
        tiling_list = re.findall(r'\d+', self.hparams.featuremap_tiling_param)
        item_num = [self.hparams.featuremap_splits_item_y, self.hparams.featuremap_splits_item_x]
        sram_size = self.hparams.featuremap_splits_sram_size
        concat_block = self.hparams.featuremap_splits_concat_blk
        ratio = self.hparams.featuremap_splits_overlap_rate / 100
        if item_num[0] * item_num[1] != 1 or sram_size != 0 or len(tiling_list):
            featuremap_partition_for_data_parallel(self.g.quantgraph,
                                                   item_num,
                                                   ratio,
                                                   sram_size,
                                                   tiling_list,
                                                   concat_block)
            # self.g.quantgraph.featuremap_partition_for_data_parallel(
            #     item_num, ratio, sram_size, tiling_list, concat_block)
            # topological_sort maybe hold,so ignore it temporarily, but it doesn't affect result
            # self.g.quantraph.topological_sort()
        ################################################################
        # delete useless op for lib
        from queue import Queue
        deleted_nodes = set()
        for n in self.g.quantgraph.nodes:
            local_deleted_tensors_count = 0
            local_deleted_nodes = Queue(maxsize=-1)
            # find need_deleted op
            for outp in n.outputs:
                if outp.need_deleted:
                    local_deleted_tensors_count += 1
                    for child in n.children:
                        for inp in child.inputs:
                            if inp.name == outp.name:
                                child.remove_input(inp)
                                break
            if local_deleted_tensors_count == len(n.outputs):
                local_deleted_nodes.put(n)
                deleted_nodes.add(n)

            # recursively delete useless op upward
            while not local_deleted_nodes.empty():
                node = local_deleted_nodes.get()
                for parent in node.parents:
                    parent_delete_count = 0
                    for child in parent.children:
                        if child in deleted_nodes:
                            parent_delete_count += 1
                    if parent_delete_count == len(parent.children):
                        local_deleted_nodes.put(parent)
                        deleted_nodes.add(parent)
        for node in deleted_nodes:
            self.g.quantgraph.remove_node(node)

        self.graph_optimize_stage3_flag = True

    # optimizer quantize has two step: collect statistic data and quantize each op
    @opt_workflow_register
    def statistic(self, refresh=False):
        """get statistic info from 2 paths:
            1 statistic file
            2 use optimizer to statistic
        """

        if os.path.exists(self.hparams.statistic_file):
            # load statistic info from file
            self.g.load_statistic_info(self.hparams.statistic_file)
        # elif self.ts_min_file=='' or self.ts_max_file=='':
        else:
            if self.calibration_dataloader is not None:
                dataloader = self.calibration_dataloader
                self.g.current_batch_size = dataloader.batch_size
                current_batch_idx = 0
                with tqdm(dataloader, desc='statistic batch', file=sys.stdout) as pbar:
                    for i, sample in enumerate(pbar):
                        inp, _ = sample
                        self.g.current_batch_idx = current_batch_idx
                        current_batch_idx += 1
                        if current_batch_idx * dataloader.batch_size > len(dataloader.dataset):
                            self.g.current_batch_size = len(dataloader.dataset) - \
                                (current_batch_idx - 1) * dataloader.batch_size
                        self.g.statistic(inp, self.hparams)

                    pbar.refresh()
            else:  # use all zeros data for statistic
                OPT_INFO(f"Optimizer will use all zeros inputs to statistic tensor information because the config is"
                         f" not setted 'calibration_data'.")
                inputs = []
                for inp in self.g.input_tensors:
                    shape = list(inp.ir_shape)
                    dtype = inp.dtype
                    inputs.append(np.zeros(shape).astype(dtype2nptype(dtype)))
                self.g.current_batch_size = self.batch_size_in_IR
                self.g.current_batch_idx = 0
                self.g.statistic(inputs, self.hparams)

        for n in self.g.nodes:
            if n.attrs['q_strategy_weight'].lower().strip() == 'in_ir':
                if 'weights_range' in n.params:
                    weights_range = n.get_param('weights_range')
                    w = n.constants['weights']
                    w.extrema_min = weights_range[0]
                    w.extrema_max = weights_range[1]
                    if None != w.extrema_min_key_axis:
                        w.extrema_min_key_axis = w.extrema_min * torch.ones_like(w.extrema_min_key_axis)
                        w.extrema_max_key_axis = w.extrema_max * torch.ones_like(w.extrema_max_key_axis)
                if 'biases_range' in n.params:
                    biases_range = n.get_param('biases_range')
                    b = n.constants['biases']
                    b.extrema_min = biases_range[0]
                    b.extrema_max = biases_range[1]
                    if None != b.extrema_min_key_axis:
                        b.extrema_min_key_axis = b.extrema_min * torch.ones_like(b.extrema_min_key_axis)
                        b.extrema_max_key_axis = b.extrema_max * torch.ones_like(b.extrema_max_key_axis)
            if n.attrs['q_strategy_activation'].lower().strip() == 'in_ir':
                for k, t in enumerate(n.outputs):
                    if 'layer_top_range' in n.params:
                        t.extrema_min = n.get_param('layer_top_range')[k][0]
                        t.extrema_max = n.get_param('layer_top_range')[k][1]
                        if None != t.extrema_min_key_axis:
                            t.extrema_min_key_axis = t.extrema_min * torch.ones_like(t.extrema_min_key_axis)
                            t.extrema_max_key_axis = t.extrema_max * torch.ones_like(t.extrema_max_key_axis)
                    else:
                        OPT_WARN(
                            f"layer_id={n.attrs['layer_id']},layer_type={n.type},output_tensor={k} has no attrs['range'] when using 'in_ir'.")
        # else:
        #     # to use min/max file and can not apply calibration strategy cause lacking information
        #     # will be deprecated soon, use statistic_file instead.
        #     self.g.load_min_max_info(self.ts_min_file, self.ts_max_file)
        #     self.hparams.calibration_strategy_for_activation = None
        #     self.hparams.calibration_strategy_for_weight = None

    @opt_workflow_register
    def quantize(self):
        """OptMaster::quantize() will use the QuantizeGraph::quantize() to quantize each op"""

        abatches = self.hparams.mixed_precision_auto_search_batches
        if abatches > 0:
            autosearch_enginer = NaiveAutoSearchMixedPrecision(self.g,
                                                               self.validation_dataloader,
                                                               self.f_metrics,
                                                               self.q_metrics,
                                                               self.hparams)
            autosearch_enginer.auto_search()

        # this pass will insert cast/quantize/dequantize op which meets the requirement
        insert_obj = [InsertQuantizeOp, InsertDeQuantizeOp, InsertCastOp]
        insert_op_pass(self.g, self.hparams, insert_obj)
        # it's not necessary to hold the original graph when self.dataloader4debug is none (cosine similarity and metric are skipped) to save memory
        self.g.quantgraph = self.g.clone() if self.dataloader4debug is not None else self.g
        unify_scales_for_multi_inputs_op_pass(self.g.quantgraph, self.hparams)
        adapt_float_subgraph_pass(self.g.quantgraph, self.hparams)
        self.g.quantize()

    def enable_fake_quant_scopes_for_debug(self, fake_quant_scopes):
        #################################################################################
        if len(fake_quant_scopes):
            # reset flags
            for n in self.g.quantgraph.nodes:
                n.attrs['debug_fake_quantize'] = False
                for t in n.outputs:
                    t.debug_flag = 0

            dmsg = 'These layers: \n'
            nset = {}
            for scope in fake_quant_scopes:
                for id in range(scope[0], scope[-1] + 1):
                    for n in self.g.quantgraph.nodes:
                        if int(n.attrs['layer_id']) == id:
                            n.attrs['debug_fake_quantize'] = True
                            msg = '(%s,%s)' % (str(n.attrs['layer_id']), str(n.type))
                            nset[msg] = True
            dmsg += '%s\nare set with debug_fake_quantize = True' % (str(sorted(nset.keys())),)
            OPT_INFO(dmsg)
        #################################################################################

    @opt_workflow_register
    def metric(self, with_float=True):
        if self.hparams.run_mode in ['quant_ir_forward']:
            self.quant_metric(self.g, self.validation_dataloader, self.q_metrics)
        elif self.hparams.run_mode in ['float_ir_forward']:
            self.float_metric(self.g, self.validation_dataloader, self.f_metrics)
        else:
            self.quant_metric(self.g, self.validation_dataloader, self.q_metrics)
            if with_float:
                self.float_metric(self.g, self.validation_dataloader, self.f_metrics)

    @opt_workflow_register
    def serialize(self, name="graph"):
        qg = self.g.quantgraph
        # first do a forward to check the final graph before serialize it
        inputs = []
        for inp in qg.input_tensors:
            shape = list(inp.ir_shape)
            dtype = inp.dtype
            inputs.append(np.zeros(shape).astype(dtype2nptype(dtype)))
        OPT_INFO("check the final graph by forwarding one sample filled with zeros")
        qg.current_batch_size = self.batch_size_in_IR
        qg.current_batch_idx = 0
        qg.forward(inputs, disable_pbar=False)
        # then serialize
        if opt_use_cuda():
            torch.cuda.empty_cache()
        OPT_INFO('Begin to serialzie IR')
        qg.serialize(name + ".txt", name + ".bin")

        # currently no need, because it's better to be hanled by GUI.
        # write opt_config_file for debug usage or internal development usage
        user_interactive_properties = [
            'q_mode_activation',
            'q_mode_weight',
            'q_bits_activation',
            'q_bits_weight',
            'q_bits_bias',
            'q_strategy_activation',
            'q_strategy_weight',
            'lut_items_in_bits',
            'force_dtype_int',
            'force_shift_positive',
            'running_statistic_momentum',
            'histc_bins',
            'scaling_bits',
        ]
        node_attrs = {}
        for node in qg.nodes:
            node_attrs[node.name] = {}
            similarity_str = ''
            if node.type in [OpType.Convolution, ]:
                node_attrs[node.name]['with_winograd'] = node.attrs['with_winograd']
            for t in node.outputs:
                node.attrs['quantization_info'][t.name]['similarity'] = t.similarity
                similarity_str += str(t.similarity) + ', '
            for k, v in node.attrs.items():
                if k in user_interactive_properties:
                    node_attrs[node.name][k] = v
            node_attrs[node.name]['just_for_display'] = {}
            node_attrs[node.name]['just_for_display']['quantization_info'] = str(node.attrs['quantization_info'])
            node_attrs[node.name]['just_for_display']['optimization_info'] = str(node.attrs['optimization_info'])
            node_attrs[node.name]['just_for_display'][
                'brief_info'] = 'layer_id = %s, layer_type = %s, similarity=%s' % (
                node.attrs['layer_id'], str(node.type), similarity_str)
        opt_config_file = os.path.join(self.hparams.output_dir, "%s_%s" %
                                       (self.hparams.model_name, DEFAULT_CONFIG_FILE))
        make_path(opt_config_file)
        with open(opt_config_file, 'w') as fw:
            json.dump(node_attrs, fw, indent=4, sort_keys=False)

        OPT_INFO('IR has been saved into ' + os.path.dirname(name))

    def report(self):
        report_dict = {}
        out_scale = []
        inp_scale = []
        if self.g.quantgraph is not None:
            for inp in self.g.quantgraph.input_tensors:
                inp_scale.append(inp.scale)
            for out in self.g.quantgraph.output_tensors:
                out_scale.append(out.scale)
            report_dict.update({'scale': {'out': out_scale, 'in': inp_scale}})

        if self.validation_dataloader is not None:
            fp_ms = []
            qt_ms = []
            metric_log = {}
            if self.f_metrics is not None and self.hparams.run_mode not in ['quant_ir_forward', 'mixed_ir_forward']:
                for fm in self.f_metrics:
                    fscore = fm.compute()
                    fp_ms.append(fscore)
                metric_log.update({'float': fp_ms})
            if self.q_metrics is not None and self.g.quantgraph is not None:
                for qm in self.q_metrics:
                    qscore = qm.compute()
                    qt_ms.append(qscore)
                metric_log.update({'quant': qt_ms})
            if len(fp_ms) and len(qt_ms):
                acc_drop = [np.array(fs) - np.array(qs) for fs, qs in zip(fp_ms, qt_ms)]
                metric_log.update({'drop': acc_drop})

            report_dict.update({'metrics': metric_log})

        elif self.dataloader4debug is not None:
            coses = []
            for ot in self.g.quantgraph.output_tensors:
                coses.append(ot.similarity)
            report_dict.update({'output tensors cosine': coses})

        if self.hparams.record_debug_acc_info:
            # just for debug
            save_fn = os.path.join(self.hparams.output_dir, self.hparams.model_name + '_record_model_acc.txt')
            if not os.path.exists(save_fn):
                save_fn = make_path(save_fn)
            with open(save_fn, 'a+') as fw:
                import time
                wtime = '\n' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n'
                cfg_info = f"[model name]:{self.hparams.model_name}, " \
                           f"[quantization method for weight]:{self.hparams.quantize_method_for_weight}," \
                           f"[quantization method for activation]:{self.hparams.quantize_method_for_activation}," \
                           f"[calibation strategy for weight]:{self.hparams.calibration_strategy_for_weight}," \
                           f"[calibation strategy for activation]:{self.hparams.calibration_strategy_for_activation}, " \
                           f"[running_statistic_momentum]:{self.hparams.running_statistic_momentum}, " \
                           f"[calibration_batch_size]:{self.hparams.calibration_batch_size}, " \
                           f"[quantization precision]: activation_bits={self.hparams.activation_bits}, " \
                           f"weight_bits={self.hparams.weight_bits}, " \
                           f"bias_bits={self.hparams.bias_bits}, " \
                           f"lut_items_in_bits={self.hparams.lut_items_in_bits},"
                fw.write(wtime)
                fw.write(cfg_info + '\n')
                fw.write(str(report_dict))

        return report_dict

    @opt_workflow_register
    def dump(self, dump_list=None, dump_attrs=None):
        """
        dump (float graph and quant graph) all nodes' tensors(constants, outputs, placeholders)
        """

        def _dump(node, md_name, path, enable_inputs=False, enable_outputs=False, enable_constants=False,
                  enable_placeholder=False, enable_params=False):
            statvfs = os.statvfs(path)
            fn_max_len = statvfs.f_namemax

            op_type = str(node.type)[7:]
            layer_name = '[' + node.name + ']'

            def _get_fn(key, tensor_name=None):
                fn = ('_'.join([md_name, op_type, layer_name, key, tensor_name])).replace('/', '_').replace(':', '_')
                return fn[:(fn_max_len - 5)]

            if enable_inputs:
                for i, inpt in enumerate(node.inputs):
                    fn = _get_fn('i' + str(i), inpt.name)
                    np.save(os.path.join(path, fn), inpt.betensor.cpu().numpy().astype(dtype2nptype(inpt.dtype)))
                    # inpt.betensor.cpu().numpy().astype(dtype2nptype(inpt.dtype)).tofile(os.path.join(path, fn) +  '_layer_' + n.attrs['layer_id'] + '.bin')
            if enable_outputs:
                for i, outt in enumerate(node.outputs):
                    fn = _get_fn('o' + str(i), outt.name)
                    np.save(os.path.join(path, fn), outt.betensor.cpu().numpy().astype(dtype2nptype(outt.dtype)))
                    # outt.betensor.cpu().numpy().astype(dtype2nptype(outt.dtype)).tofile(os.path.join(path, fn) + '_layer_' + n.attrs['layer_id'] + '.bin')
            if enable_constants:
                _brev_map = {
                    "weights": 'w',
                    "biases": "b",
                }
                for name, consts in node.constants.items():
                    n = _brev_map.get(name, name)
                    fn = _get_fn(n, consts.name)
                    np.save(os.path.join(path, fn), consts.betensor.cpu().numpy().astype(dtype2nptype(consts.dtype)))
                    # consts.betensor.cpu().numpy().astype(dtype2nptype(consts.dtype)).tofile(os.path.join(path, fn) +  '_layer_' + n.attrs['layer_id'] + '.bin')
            if enable_placeholder:
                for i, plt in enumerate(node.placeholders):
                    fn = _get_fn('p' + str(i), plt.name)
                    np.save(os.path.join(path, fn), plt.betensor.cpu().numpy())
            if enable_params:
                # TODO
                pass

        _dump_inputs = functools.partial(_dump, enable_inputs=True)
        _dump_outputs = functools.partial(_dump, enable_outputs=True)
        _dump_placeholders = functools.partial(_dump, enable_placeholders=True)
        _dump_constants = functools.partial(_dump, enable_constants=True)
        _dump_params = functools.partial(_dump, enable_params=True)
        _dump_all = functools.partial(_dump, enable_inputs=True, enable_outputs=True,
                                      enable_constants=True, enable_params=True)

        _attrs_2_dump = {
            'input': _dump_inputs,
            'output': _dump_outputs,
            'constant': _dump_constants,
            'placeholder': _dump_placeholders,
            'params': _dump_params,
            'all': _dump_all,
        }

        if dump_list == None:
            # default to dump all node
            dump_list = []
            for n in self.g.nodes:
                dump_list.append(int(n.attrs['layer_id']))

        if dump_attrs == None:
            # default to dump all attributes: inputs, outputs, constants, placeholder tensor data
            dump_attrs = ['all']

        bp_fp = os.path.join(self.hparams.dump_dir, 'float32')
        bp_qt = os.path.join(self.hparams.dump_dir, 'quant')
        make_dir_path(bp_fp)
        make_dir_path(bp_qt)

        for n in self.g.nodes:
            layer_id = n.attrs['layer_id']
            if int(layer_id) in dump_list:
                for attr in dump_attrs:
                    _attrs_2_dump[attr](n, self.hparams.model_name, bp_fp)
                # OPT_DEBUG('Float32 dump layer_id:%s has done.' % (layer_id))
        for n in self.g.quantgraph.nodes:
            layer_id = n.attrs['layer_id']
            if int(layer_id) in dump_list:
                for attr in dump_attrs:
                    _attrs_2_dump[attr](n, self.hparams.model_name, bp_qt)
                # OPT_DEBUG('Quant dump layer_id:%s has done.' % (layer_id))
        OPT_INFO("Optimizer dumps Done, please check the file in %s and %s path" % (bp_fp, bp_qt))

    def collect_information_for_debug(self):
        '''
        mainly output the debug information:
        -enable fake quantize
        -collect similarity between float and int
        -collect op running cost time
        -dump tensor values
        :return:
        '''
        if self.dataloader4debug != None:
            # set debug_fake_quantize flag according to hparams
            self.enable_fake_quant_scopes_for_debug(self.fake_quant_scopes)
            #################################################################################
            # #for internal debug usage
            # dflag = True
            # fake_quant_scopes = []
            # for n in self.g.quantgraph.nodes :
            #     if n.type in [OpType.LayerNorm, ] :
            #         lid = int(n.attrs['layer_id'])
            #         fake_quant_scopes.append((lid, lid))
            # if dflag :
            #     self.enable_fake_quant_scopes_for_debug(fake_quant_scopes)
            #################################################################################
            # log per layer similarity info after quantization

            # enable to calculate op running time
            for n in self.g.nodes:
                n.attrs['calculate_running_time'] = True
            for n in self.g.quantgraph.nodes:
                n.attrs['calculate_running_time'] = True

            self.g.current_batch_size = self.dataloader4debug.batch_size
            self.g.quantgraph.current_batch_size = self.dataloader4debug.batch_size
            check_sim_len = self.hparams.similarity_data_num if self.hparams.similarity_data_num <= len(
                self.dataloader4debug) else 1
            OPT_INFO(
                f'collecting per-layer similarity infomation between float graph and quanted graph by forwarding {check_sim_len} sample on both of them')
            for i, sample in zip(range(check_sim_len), self.dataloader4debug):
                inp, _ = sample
                self.g.current_batch_idx = i
                self.g.quantgraph.current_batch_idx = i
                if (i + 1) * self.dataloader4debug.batch_size > len(self.dataloader4debug.dataset):
                    bsize = len(self.dataloader4debug.dataset) - i * self.dataloader4debug.batch_size
                    self.g.current_batch_size = bsize
                    self.g.quantgraph.current_batch_size = bsize
                check_nodes_similarity(self.g, self.g.quantgraph, inp, keep_tensors=self.hparams.dump)
            show_similarity(self.g.quantgraph)

            calculate_op_running_time(self.g, self.g.quantgraph)

            # dump tensor
            if self.hparams.dump:
                self.dump()

            if opt_use_cuda():
                torch.cuda.empty_cache()

    @opt_workflow_register
    def float_metric(self, graph, dataloader, fmetrics):
        if opt_use_cuda():
            torch.cuda.empty_cache()
        graph_inference(graph, graph.forward, dataloader, fmetrics, with_float=True)
        for fmetric in fmetrics:
            OPT_INFO('float metric: %s' % (fmetric.report()))

    @opt_workflow_register
    def quant_metric(self, graph, dataloader, qmetrics):
        if opt_use_cuda():
            torch.cuda.empty_cache()
        graph_inference(graph.quantgraph, graph.qforward, dataloader, qmetrics)
        for qmetric in qmetrics:
            OPT_INFO('quant metric: %s' % (qmetric.report()))

    def open_quantized_flag(self):
        if self.g.quantgraph is None:
            OPT_ERROR(f"open node's quantized flag, please create the quantgraph firstly.")
            return None
        for node in self.g.quantgraph.nodes:
            node.quantized = True

    def close_quantized_flag(self):
        if self.g.quantgraph is None:
            OPT_ERROR(f"open node's quantized flag, please create the quantgraph firstly.")
            return None
        for node in self.g.quantgraph.nodes:
            node.quantized = False

    def deduce_quantization_infos(self, graph):
        def _deduce_quantization_info_to_tensor_from_ir(node, updated_fields):
            in_out_tensors = [*node.inputs, *node.outputs]
            for t in in_out_tensors:
                if is_float(t.dtype):
                    continue
                o_dtype = t.dtype
                qmin, qmax = dtype2range(o_dtype)
                qbits = dtype2bits(o_dtype)
                quantization_infos = {
                    'qmin': qmin,
                    'qmax': qmax,
                    'qbits': qbits
                }
                for field in updated_fields:
                    if field in quantization_infos.keys():
                        t.__setattr__(field, quantization_infos[field])

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
            else:
                dtypes = [t.dtype for t in (list(node.outputs) + list(node.constants.values())) + list(node.inputs)]
                for dt in dtypes:
                    if is_float(dt):
                        node.quantized = False
                        break
            if node.quantized:
                _deduce_quantization_info_to_tensor_from_ir(node, get_tensor_default_property())
        return graph

    def run_quant_ir_forward(self):
        if self.g.quantgraph is None:
            self.g.quantgraph = self.g.clone()
        self.deduce_quantization_infos(self.g.quantgraph)
        # self.open_quantized_flag()
        self.quant_metric(self.g, self.validation_dataloader, self.q_metrics)

    def run_float_ir_forward(self):
        self.float_metric(self.g, self.validation_dataloader, self.f_metrics)

    def run_mixed_ir_forward(self):
        if self.g.quantgraph is None:
            self.g.quantgraph = self.g.clone()
        self.deduce_quantization_infos(self.g.quantgraph)
        self.quant_metric(self.g, self.validation_dataloader, self.q_metrics)

    def run_default(self):
        # self.prepare(self.hparams)
        self.optimize()
        self.serialize(os.path.join(self.hparams.output_dir, self.hparams.quant_ir_name))
        if self.validation_dataloader is not None:
            self.metric(self.hparams.eval_original_model)

    def __call__(self, *args, **kwargs):
        self.prepare(self.hparams)
        if hasattr(self.g, 'compat_quantized_model') and self.g.compat_quantized_model:
            OPT_INFO(f"Now we do quantization transform in Optimizer.")
            try:
                from AIPUBuilder.core import quantize_transform as qtlib_quantize_transform
            except Exception as e:
                OPT_ERROR(
                    f"The AIPUBuilder.core module is required when compat_quantized_model is True. now error message: {e}")

            new_quantization_method_ops_type = []
            if self.hparams.compat_quantized_model_ops != '':
                lower_optype = {}
                for k, v in OpType.__dict__.items():
                    lower_optype.update({k.lower(): v})
                for op in self.hparams.compat_quantized_model_ops.strip().replace(' ', '').split(','):
                    new_quantization_method_ops_type.append(lower_optype[op])
            # pre-pass
            convert_resize_to_convolution(self.g)
            for n in self.g.nodes:
                n.attrs['unify_shifts_mode'] = self.hparams.compat_quantized_model_unify_shifts_mode
                # now qat model fixed to 13bits, TODO: set by cfg fields and distinguish with opt flow.
                n.attrs['multiplier_bits'] = 13
                if 'conv_from_resize_opt' not in n.attrs:
                    n.attrs['trigger_float_op'] = 'float16_preferred' if self.hparams.trigger_float_op.lower(
                    ) == 'disable' else self.hparams.trigger_float_op.lower()
                if self.hparams.compat_quantized_model_int8_to_uint8:
                    n.attrs["int8_to_uint8"] = True
                if n.type == OpType.Constant:
                    n.attrs['scale_zp_need_quantize'] = True
                if n.type in new_quantization_method_ops_type:
                    n.attrs['tflite_quantization'] = True
                if n.type == OpType.Eltwise:
                    # n.attrs['eltwise_quantization'] = True
                    # n.attrs['eltwise_quantization'] = False
                    n.attrs['left_shift_bits'] = self.hparams.compat_quantized_model_left_shift_bits
                if n.type == OpType.BasicLSTM:
                    n.attrs['weight_dim'] = 1
                    n.attrs['set_default_placeholder_info'] = True
                    # n.attrs['start_basic_lstm_id'] = 64
                    # n.attrs['start_basic_lstm_id'] = 32
                if n.type == OpType.Cast:
                    n.attrs["eliminate_cast"] = self.hparams.compat_quantized_model_eliminate_cast

            cg = convert_opt_graph_to_aipu_graph(self.g)
            qtlib_quantize_transform(cg, run_mode=self.hparams.compat_quantized_model_strategy)
            name = os.path.join(self.hparams.output_dir, self.hparams.quant_ir_name)
            if not os.path.exists(self.hparams.output_dir):
                os.makedirs(self.hparams.output_dir)
            if self.hparams.run_mode not in ['quant_ir_forward', ]:
                cg.attrs['serialize_scale_zp'] = True
                cg.serialize(f"{name}.txt", f"{name}.bin")
                return
            else:
                self.g.quantgraph = convert_aipu_graph_to_opt_graph(cg)
                self.g.quantgraph.serialize(f"{name}.txt", f"{name}.bin")

        if self.hparams.run_mode == 'float_ir_forward':
            OPT_INFO(f"Now configure the Float Compass IR, and Optimizer uses this IR only to float inference.")
            self.run_float_ir_forward()
        elif self.hparams.run_mode == 'quant_ir_forward':
            OPT_INFO(
                f"Now configure the Quantization Compass IR, and Optimizer uses this IR only to quantization inference.")
            self.run_quant_ir_forward()
        elif self.hparams.run_mode == 'mixed_ir_forward':
            OPT_INFO(
                f"Now configure the Mixed-Float-Quantization Compass IR, and Optimizer uses this IR only to inference.")
            self.run_mixed_ir_forward()
        else:
            self.run_default()
        report = self.report()
        return report
