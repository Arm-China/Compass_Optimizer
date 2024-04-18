# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import QuantMode, passes_run
import torch


@passes_run
def split_matmul(graph, config):
    """
    when matmul uses the activation perchannel quantization, the matmul will generate a matrix doscale/doshift. this
    pass will split the matmul to matmul + constant + mul pattern.
    """

    for n in graph.nodes:
        if n.type != OpType.MatMul:
            continue

        q_mode_activation = n.attrs['q_mode_activation']
        if not QuantMode.is_per_channel(q_mode_activation):
            continue

        if not QuantMode.is_per_channel(n.inputs[0].pnode.attrs['q_mode_activation']):
            # if the parent node of input0 is not per-channel activation quantization,
            # the matrix doscale/doshift doenot exist.
            continue

        # create the constant node and mul node
        matmul_out_t = n.outputs[0]
        matmul_children = n.children
        const_weight_shape = [1 if inpt.key_axis is None else inpt.ir_shape[inpt.key_axis] for inpt in n.inputs]
        const_weight_data = torch.ones(const_weight_shape, device=matmul_out_t.device).float()

        const_node = PyNode(graph.get_valid_node_name(f"{n.name}_constant"), OpType.Constant)
        const_out_t = PyTensor(graph.get_valid_tensor_name(f"{matmul_out_t.name}_const"), TensorShape(const_weight_shape),
                               dtype=matmul_out_t.dtype)
        const_weight_t = PyTensor(graph.get_valid_tensor_name(f"{const_node}_weights"), const_weight_data)
        const_node.constants['weights'] = const_weight_t
        const_node.add_output(const_out_t)

        # set const node attr
        const_node.attrs.update(n.attrs.clone())
        # now defaultly uses per-tensor quantization for this inserted constant op
        # const_node.attrs['q_mode_activation'] = QuantMode.to_per_tensor(n.attrs['q_mode_activation'])
        # const_node.attrs['q_mode_weight'] = QuantMode.to_per_tensor(n.attrs['q_mode_activation'])
        const_node.attrs['layer_id'] = '0' + str(n.attrs['layer_id'])

        mul_node = PyNode(graph.get_valid_node_name(f"{n.name}_mul"), OpType.Mul)
        # mul_out_t = matmul_out_t.clone(graph.get_valid_tensor_name(f"{matmul_out_t.name}_mul"))
        mul_out_t = PyTensor(graph.get_valid_tensor_name(f"{matmul_out_t.name}_mul"), matmul_out_t.ir_shape,
                             dtype=matmul_out_t.dtype)

        mul_out_t.key_axis = matmul_out_t.key_axis

        mul_node.add_input(matmul_out_t)
        mul_node.add_input(const_out_t)
        mul_node.add_output(mul_out_t)

        mul_node.attrs.update(n.attrs.clone())
        mul_node.attrs['layer_id'] = '0' + const_node.attrs['layer_id']

        for mc in matmul_children:
            idx = mc.remove_input(matmul_out_t)
            mc.add_input(mul_out_t, idx)

        graph.add_node(const_node)
        graph.add_node(mul_node)

        n.attrs['splitted_matmul'] = True
