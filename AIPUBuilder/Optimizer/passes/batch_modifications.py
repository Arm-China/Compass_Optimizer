# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
from copy import deepcopy


def modify_batch_dim(graph: PyGraph, config):
    batch = config.modify_batch_dim
    origin = graph.input_tensors[0].ir_shape[config.data_batch_dim]

    for node in graph.nodes:
        if node.type in [OpType.Constant]:
            continue
        for output in node.outputs:
            if output.ir_shape[0] == origin:
                output.ir_shape = TensorShape((batch, ) + output.ir_shape[1:])
            else:
                new_shape = list(node.outputs[0].ir_shape)
                bs = new_shape[0] // origin * batch
                new_shape[0] = bs
                node.outputs[0].ir_shape = TensorShape(new_shape)
        if node.type == OpType.Reshape:
            node.params['shape'] = list(node.outputs[0].ir_shape)
        if node.type == OpType.Slice:
            begin = node.params['begin']
            end = node.params['end']
            full_shape = node.inputs[0].ir_shape
            begin = [min(full_shape[i], begin[i]) for i in range(len(full_shape))]
            end = [min(full_shape[i], end[i]) for i in range(len(full_shape))]
            node.params['begin'] = begin
            node.params['end'] = end


def copy_parallel_batch(graph: PyGraph, config):
    parallel = config.export_parallel_batch
    origin_batch = graph.input_tensors[0].ir_shape[config.data_batch_dim]

    outputs = [[graph.output_tensors[i]] for i in range(len(graph.output_tensors))]

    tensor_map = {}

    for batch in range(parallel - 1):
        OPT_INFO(f"Copying parallel batch {batch + 1} / {parallel - 1}")
        for node in list(graph.nodes)[:]:
            if node.type == OpType.Input:
                t = node.outputs[0].clone()
                node.add_output(t)
                tensor_map[node.outputs[0]] = t
                continue
            cloned = PyNode(node.name+f"_clone_{batch}", node.type)
            for k, v in node.params.items():
                cloned.params[k] = deepcopy(v)
            for k, v in node.constants.items():
                cloned.constants[k] = v.clone()
            graph.add_node(cloned)
            for inp in node.inputs:
                cloned.add_input(tensor_map[inp])
            for out in node.outputs:
                cout = out.clone()
                tensor_map[out] = cout
                cloned.add_output(cout)
        for i, out in enumerate(graph.output_tensors):
            outputs[i].append(tensor_map[out])

    input_tensors = []
    for i in range(len(graph.input_tensors)):
        inp = graph.input_tensors[i]
        node = inp.pnode
        node.type = OpType.Split
        node.name += "_Split"
        node.params['axis'] = config.data_batch_dim
        node.params['splits'] = [origin_batch for _ in range(parallel)]
        inp_t = inp.clone(inp.name + "_input")
        shape = list(inp_t.ir_shape)
        shape[config.data_batch_dim] = origin_batch * parallel
        inp_t.ir_shape = TensorShape(shape)
        inp_n = PyNode(node.name + "_Input", OpType.Input)
        inp_t.pnode = inp_n
        inp_n.add_output(inp_t)
        node.add_input(inp_t)
        graph.nodes = [inp_n] + graph.nodes
        input_tensors.append(inp_t)

    graph.input_tensors = input_tensors

    output_tensors = []
    for i, out in enumerate(graph.output_tensors):
        concat = PyNode(out.name, OpType.Concat)
        graph.add_node(concat)
        concat.params['axis'] = config.data_batch_dim
        for cout in outputs[i]:
            concat.add_input(cout)
        if out.pnode.quantized:
            concat.quantized = True
            concat.params['unquantifiable'] = False
            concat.params['scale_value'] = [1 for i in range(parallel)]
            concat.params['scale_type'] = [Dtype.UINT8 for i in range(parallel)]
            concat.params['shift_value'] = [0 for i in range(parallel)]
            concat.params['shift_type'] = [Dtype.UINT8 for i in range(parallel)]
        else:
            concat.quantized = False
            concat.params['unquantifiable'] = True
        concat_t = out.clone(out.name + "_concat")
        shape = list(concat_t.ir_shape)
        shape[config.data_batch_dim] = origin_batch * parallel
        concat_t.ir_shape = TensorShape(shape)
        concat.add_output(concat_t)
        output_tensors.append(concat.outputs[0])

    graph.output_tensors = output_tensors
