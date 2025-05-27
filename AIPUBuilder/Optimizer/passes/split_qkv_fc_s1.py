# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *


def split_qkv_fc(graph, config):
    if not config.enable_pass_split_qkv_fc_in_transformer:
        return

    for i, n in enumerate(graph.nodes):
        if n.type != OpType.FullyConnected:
            continue
        neednot_quant_op = [OpType.Reshape]
        # create the constant node and mul node
        fc_out_t = n.outputs[0]
        fc_children_split = n.children[0]
        while fc_children_split.type in neednot_quant_op:
            if len(fc_children_split.children) > 0:
                fc_children_split = fc_children_split.children[0]
            else:
                break
        if fc_children_split.type != OpType.Split:
            continue

        split_num = len(fc_children_split.outputs)

        const_weight_shape = n.constants['weights'].ir_shape
        row_num = const_weight_shape[0]//split_num
        add_node_list = []
        for i in range(split_num):

            fc = n.clone(graph.get_valid_node_name(n.name+str(i)))
            fc.constants['weights'].ir_shape = row_num, const_weight_shape[1]
            fc.constants['biases'].ir_shape = row_num,
            fc.constants['weights'].betensor = n.constants['weights'].betensor[i*row_num:(i+1)*row_num, :]
            fc.constants['biases'].betensor = n.constants['biases'].betensor[i*row_num:(i+1)*row_num]

            fc.attrs['layer_id'] = str('0' * (i+1)) + n.attrs['layer_id']
            fc.params['num_output'] = row_num

            fc.remove_input(fc.inputs[0])
            fc.add_input(n.parents[0].outputs[0])
            fc.remove_output(fc.outputs[0])
            fc.add_output(fc_children_split.children[i].inputs[0])
            fc.outputs[0].ir_shape = fc_out_t.ir_shape[0], row_num

            add_node_list.append(fc)

        graph.remove_node(fc_children_split)

        graph.remove_node(n.children[0])
        graph.remove_node(n)
        for i in range(split_num):
            graph.add_node(add_node_list[i])
