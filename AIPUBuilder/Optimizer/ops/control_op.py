# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import torch

register_optype('If')
register_optype('Loop')


@op_register(OpType.If)
def control_if_forward(self, *args):
    then_branch_g = self.get_sub_graph('then_branch')
    else_branch_g = self.get_sub_graph('else_branch')
    then_branch_inputs_num = self.params['then_branch_inputs_num']
    else_branch_inputs_num = self.params['else_branch_inputs_num']
    cond = self.inputs[0].betensor
    cg = None
    feed_data = []
    if cond:
        cg = then_branch_g
        for t in self.inputs[1:(1+then_branch_inputs_num)]:
            feed_data.append(t.betensor.clone())
    else:
        cg = else_branch_g
        for t in self.inputs[(1+then_branch_inputs_num):]:
            feed_data.append(t.betensor.clone())
    if self.quantized:
        cg = cg.quantgraph
    cg.forward(feed_data, disable_pbar=True, keep_tensors=True)

    for ot, st in zip(self.outputs, cg.output_tensors):
        ot.betensor = st.betensor.clone()

    if not self.quantized:
        if 'subgraph_constants' not in self.attrs:
            self.attrs['subgraph_constants'] = []
            for sn in then_branch_g.nodes + else_branch_g.nodes:
                # push sub_graph's activation tensors into self.placeholders for statistic
                self.placeholders.extend(list(sn.outputs) + list(sn.placeholders))
                # push sub_graph's constant tensors into self.attrs for statistic
                self.attrs['subgraph_constants'].extend(
                    list(sn.constants.values()) + sn.get_attrs('subgraph_constants', optional=True, default_value=[]))


# currently not support per-layer configuration for subgraphs
g_q_attrs_list = ['trigger_float_op', 'q_mode_activation', 'q_mode_weight', 'q_mode_bias', 'q_bits_activation', 'q_bits_weight', 'q_bits_bias', 'lut_items_in_bits', 'multiplier_bits',
                  'bias_effective_bits', 'unify_shifts_for_aiff', 'weight_block_size', 'force_dtype_int', 'force_shift_positive', 'min_compatible_zhouyi_target', 'batch_size_in_IR']


def subg_inp_quantize(self, *args):
    out = self.outputs[0]
    qinfo = out.attrs['qinfo_from_call_func']
    out.scale = qinfo['scale']
    out.zerop = qinfo['zerop']
    out.qbits = qinfo['qbits']
    out.dtype = qinfo['dtype']
    out.qmin = qinfo['qmin']
    out.qmax = qinfo['qmax']
    out.qinvariant = qinfo['qinvariant']


@quant_register(OpType.If)
def control_if_quantize(self, *args):
    OPT_WARN(f'{self} : quantize control flow operator may cause unpredictable accuracy issues')
    then_branch_g = self.get_sub_graph('then_branch')
    else_branch_g = self.get_sub_graph('else_branch')
    then_branch_inputs_num = self.params['then_branch_inputs_num']
    else_branch_inputs_num = self.params['else_branch_inputs_num']
    for sn in then_branch_g.nodes + else_branch_g.nodes:
        sn.params['unquantifiable'] = self.params['unquantifiable']
        for qv in g_q_attrs_list:
            sn.attrs[qv] = self.attrs[qv]

    for i, t in enumerate(then_branch_g.input_tensors):
        rt = self.inputs[1+i]
        t.attrs['qinfo_from_call_func'] = {'scale': rt.scale, 'zerop': rt.zerop, 'qbits': rt.qbits,
                                           'dtype': rt.dtype, 'qmin': rt.qmin, 'qmax': rt.qmax, 'qinvariant': rt.qinvariant}
    for i, t in enumerate(else_branch_g.input_tensors):
        rt = self.inputs[1+then_branch_inputs_num+i]
        t.attrs['qinfo_from_call_func'] = {'scale': rt.scale, 'zerop': rt.zerop, 'qbits': rt.qbits,
                                           'dtype': rt.dtype, 'qmin': rt.qmin, 'qmax': rt.qmax, 'qinvariant': rt.qinvariant}
    # disable auto quantization of subgraph input tensors in case of overwriting
    bak_inp_quantize = QUANT_OP_DICT[OpType.Input]
    QUANT_OP_DICT[OpType.Input] = subg_inp_quantize
    then_branch_g.quantgraph = None
    then_branch_g.quantize(disable_pbar=True)
    else_branch_g.quantgraph = None
    else_branch_g.quantize(disable_pbar=True)
    QUANT_OP_DICT[OpType.Input] = bak_inp_quantize
    cond = self.inputs[0].betensor
    cg = then_branch_g if cond else else_branch_g
    for ot, st in zip(self.outputs, cg.quantgraph.output_tensors):
        ot.clone_qinfo(st)


@op_register(OpType.Loop)
def control_loop_forward(self, *args):
    cg = self.get_sub_graph('body')
    if self.quantized:
        cg = cg.quantgraph
    M = self.inputs[0].betensor
    cond_in = self.inputs[1].betensor

    N = len(self.inputs) - 2
    i = 0
    feed_data = []
    for t in self.inputs:
        feed_data.append(t.betensor.clone())
    K = len(self.outputs) - N
    scan_outputs = []
    for _ in range(K):
        scan_outputs.append([])
    while i < M and cond_in:
        cg.forward(feed_data, disable_pbar=True, keep_tensors=True)
        cond_in = cg.output_tensors[0].betensor
        feed_data = [M, cond_in] + [ot.betensor for ot in cg.output_tensors[1:1+N]]
        for k in range(K):
            scan_outputs[k].append(cg.output_tensors[1+N+k].betensor.unsqueeze(0))
        i += 1

    for k in range(N):
        self.outputs[k].betensor = cg.output_tensors[1+k].betensor.clone()
    for k in range(K):
        self.outputs[N+k].betensor = torch.cat(scan_outputs[k])

    if not self.quantized:
        if 'subgraph_constants' not in self.attrs:
            self.attrs['subgraph_constants'] = []
            for sn in cg.nodes:
                # push sub_graph's activation tensors into self.placeholders for statistic
                self.placeholders.extend(list(sn.outputs) + list(sn.placeholders))
                # push sub_graph's constant tensors into self.attrs for statistic
                self.attrs['subgraph_constants'].extend(
                    list(sn.constants.values()) + sn.get_attrs('subgraph_constants', optional=True, default_value=[]))


@quant_register(OpType.Loop)
def control_loop_quantize(self, *args):
    OPT_WARN(f'{self} : quantize control flow operator may cause unpredictable accuracy issues')
    cg = self.get_sub_graph('body')
    for sn in cg.nodes:
        sn.params['unquantifiable'] = self.params['unquantifiable']
        for qv in g_q_attrs_list:
            sn.attrs[qv] = self.attrs[qv]

    for i, t in enumerate(cg.input_tensors):
        rt = self.inputs[i]
        t.attrs['qinfo_from_call_func'] = {'scale': rt.scale, 'zerop': rt.zerop, 'qbits': rt.qbits,
                                           'dtype': rt.dtype, 'qmin': rt.qmin, 'qmax': rt.qmax, 'qinvariant': rt.qinvariant}
    # disable auto quantization of subgraph input tensors in case of overwriting
    bak_inp_quantize = QUANT_OP_DICT[OpType.Input]
    QUANT_OP_DICT[OpType.Input] = subg_inp_quantize
    cg.quantgraph = None
    cg.quantize(disable_pbar=True)
    QUANT_OP_DICT[OpType.Input] = bak_inp_quantize
    for i, t in enumerate(self.outputs):
        t.clone_qinfo(cg.quantgraph.output_tensors[1+i])
