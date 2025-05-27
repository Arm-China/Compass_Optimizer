# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.utils import *
import torch
import copy
import numpy as np


# #############################################################################
# feature map partition for multi-core
# currently, support OPs: conv,deconv,pooling, fc, matmul,d2s
# insert related op from down to up
# 1. search continious seperable OP, calculate overlap from down to up
# 2. split the top op input featuremap according to item_num, insert related slice op
# 3. insert single conv item for original each conv node
#    for i range(items)
#        for n range(nodes)
#           .........
# 4. at the bottom layer, insert related concat op
############################################################################

def featuremap_partition_for_data_parallel(graph, item_num, ratio, sram_size, tiling_list, concat_block):
    # eltwise cause some unexpected issue, so remove eltwise temporarily
    tranparent_op = [OpType.Reshape, OpType.Cast, OpType.Activation, OpType.Transpose]
    tot_nodes_num = len(graph.nodes)

    ###########################################################################################
    # analyse node,get multi-items

    def calc_work_item_size_by_output(out_size, k_size, stride, padding):
        return ((out_size - 1) * stride + k_size - padding[..., 0:2] - padding[..., 2:4])

    def calc_work_item_size_output(in_size, k_size, stride, padding):
        return ((in_size - k_size + padding[..., 0:2] + padding[..., 2:4]) // stride + 1)

    # overlap=w_len-(next_layer_w_in*stride-padding)
    def calc_work_item_overlap(w_len, next_layer_w_in, stride, padding):
        return (w_len - (next_layer_w_in * stride - padding[..., 0:2] - padding[..., 2:4]))

    # padding layout format[top,left,bottom,right]
    def fill_padding_param_for_all_items(node):

        padding_left = node.params['pad_left']
        padding_right = node.params['pad_right']
        padding_top = node.params['pad_top']
        padding_bottom = node.params['pad_bottom']
        dl_x = node.params['dilation_x']
        dl_y = node.params['dilation_y']
        stride_x = node.params['stride_x']
        kernel_x = node.params['kernel_x']
        kernel_x = (kernel_x-1)*dl_x+1
        stride_y = node.params['stride_y']
        kernel_y = node.params['kernel_y']
        kernel_y = (kernel_y-1)*dl_y+1
        if node.type == OpType.ConvTranspose:
            padding_left = kernel_x - padding_left - 1
            padding_right = kernel_x - padding_right - 1
            padding_top = kernel_y - padding_top - 1
            padding_bottom = kernel_y - padding_bottom - 1
            stride_x = 1
            stride_y = 1

        padding = torch.zeros((item_num[0], item_num[1], 4), dtype=torch.int16, device=device_type)
        w = node.inputs[0].ir_shape[2]
        # some conv padding is useless, such as input w=10, k=3,s=2,padding left=1, padding right=1
        # o = (10-3+2)/2 + 1 = 5, to get this 5 point, k window slide to indice 7,8,9, right padding is useless
        padding_right = padding_right - (w - kernel_x + padding_left + padding_right) % stride_x
        padding_right = padding_right if padding_right > 0 else 0

        h = node.inputs[0].ir_shape[1]
        padding_bottom = padding_bottom - (h - kernel_y + padding_top + padding_bottom) % stride_y
        padding_bottom = padding_bottom if padding_bottom > 0 else 0

        # padding = torch.zeros((item_num[0],item_num[1],4),dtype=torch.int16,device=device_type)
        # left top corner
        padding[0, 0, 0:2] = torch.tensor([padding_top, padding_left])
        # right bottom corner
        padding[item_num[0] - 1, item_num[1] - 1, 2:4] = torch.tensor([padding_bottom, padding_right])
        # right top corner
        padding[0, item_num[1] - 1, 3] = torch.tensor([padding_right])
        padding[0, item_num[1] - 1, 0] = torch.tensor([padding_top])
        # left bottom corner
        padding[item_num[0] - 1, 0, 2] = torch.tensor([padding_bottom])
        padding[item_num[0] - 1, 0, 1] = torch.tensor([padding_left])
        # top row
        padding[0, 1:item_num[1] - 1, 0] = torch.tensor([padding_top])
        # bottom row
        padding[item_num[0] - 1, 1:item_num[1] - 1, 2] = torch.tensor([padding_bottom])

        # left col
        padding[1:item_num[0] - 1, 0, 1] = torch.tensor([padding_left])
        # right col
        padding[1:item_num[0] - 1, item_num[1] - 1, 3] = torch.tensor([padding_right])
        stride = torch.tensor([stride_y, stride_x], device=device_type)
        return padding, stride

    def fill_deconv_padding_param_for_all_items(node, layerid, all_work_items_len, each_node_work_items_out, all_work_items_in):

        padding_left = node.params['pad_left']
        padding_right = node.params['pad_right']
        padding_top = node.params['pad_top']
        padding_bottom = node.params['pad_bottom']
        stride_x = node.params['stride_x']
        kernel_x = node.params['kernel_x']
        stride_y = node.params['stride_y']
        kernel_y = node.params['kernel_y']

        padding = torch.zeros((item_num[0], item_num[1], 4), dtype=torch.int16, device=device_type)
        h_pos = 0
        for item_y in range(item_num[0]):
            h_out = each_node_work_items_out[layerid, item_y, 0, 0].int().item()
            h = all_work_items_len[layerid, item_y, 0, 0].int().item()
            h_in = all_work_items_in[layerid, item_y, 0, 0].int().item()
            h_pos = h_pos+h_in
            padding_bottom = kernel_y - (h_out + padding_top - (h - 1) * stride_y)
            padding_left = node.params['pad_left']
            w_pos = 0
            for item_x in range(item_num[1]):
                w_out = each_node_work_items_out[layerid, item_y, item_x, 1].int().item()
                w = all_work_items_len[layerid, item_y, item_x, 1].int().item()
                w_in = all_work_items_in[layerid, item_y, item_x, 1].int().item()
                w_pos = w_in + w_pos
                # modify center's part's padding
                padding_right = kernel_x - (w_out + padding_left - (w - 1) * stride_x)
                # [top,left,bottom,right]
                padding[item_y, item_x, :] = torch.tensor(
                    [padding_top, padding_left, padding_bottom, padding_right], device=device_type)

                padding_left = kernel_x-1-(stride_x-w_pos % stride_y)*(w_pos % stride_x != 0)

            padding_top = kernel_y-1-(stride_y-h_pos % stride_y)*(h_pos % stride_y != 0)
        return padding

    ###########################################################################################
    # pooling ceil mode, if according to formula expect_out=(h-k+p)//s+1< IR's out shape

    def pooling_ceil_mode_padding(node):
        left = node.params['pad_left']
        right = node.params['pad_right']
        top = node.params['pad_top']
        bottom = node.params['pad_bottom']
        stride_x = node.params['stride_x']
        kernel_x = node.params['kernel_x']
        stride_y = node.params['stride_y']
        kernel_y = node.params['kernel_y']

        # [top,left,bottom,right]
        padding = torch.tensor([top, left, bottom, right], device=device_type)
        kernel_size = torch.tensor([kernel_y, kernel_x], device=device_type)
        stride = torch.tensor([stride_y, stride_x], device=device_type)
        hw_in = torch.tensor([node.inputs[0].ir_shape[1], node.inputs[0].ir_shape[2]], device=device_type)
        hw_out = torch.tensor([node.outputs[0].ir_shape[1], node.outputs[0].ir_shape[2]], device=device_type)
        expect_out = calc_work_item_size_output(hw_in, kernel_size, stride, padding)
        return hw_out - expect_out
        ###############################################################################################
        # calculate partition Ops input shape and output shape, neightbor part slice offset(work_items_in)
        # bottom shape work_items_len
        # top shape work_items_out
        #############################################################################################

    def get_partition_params_by_back_propagate(end_split_node_idx, start_split_node_idx):

        # calculate conv partition params by back propagate
        if not torch.is_tensor(item_num):
            v_item_num = torch.tensor(item_num, device=device_type)
        else:
            v_item_num = item_num
        nodes_num = end_split_node_idx - start_split_node_idx + 1

        last_node = graph.nodes[end_split_node_idx]

        if 'kernel_x' in last_node.params:
            fused_out = torch.tensor([last_node.outputs[0].shape[1],
                                      last_node.outputs[0].shape[2]], device=device_type)
        elif last_node.type == OpType.DepthToSpace or last_node.type in tranparent_op or last_node.type == OpType.Eltwise:
            fused_out = torch.tensor(
                [last_node.inputs[0].shape[1], last_node.inputs[0].shape[2]], device=device_type)
        # fc has one input, A[H,W]*WEIGHT[W,I]=C[H,I],W can't be splited
        elif last_node.type == OpType.FullyConnected:
            fused_out = torch.tensor([last_node.inputs[0].shape[-2],
                                      last_node.outputs[0].shape[-1]], device=device_type)
        # matmul has 2 inputs matrix A[...,H,W]*B[...,W,I]=C[...,H,I]
        elif last_node.type == OpType.MatMul:
            fused_out = torch.tensor([last_node.inputs[0].shape[-2],
                                      last_node.inputs[1].shape[-1]], device=device_type)
        work_items_len = torch.zeros((item_num[0], item_num[1], 2), device=device_type)
        work_items_out = torch.ones((item_num[0], item_num[1], 2), device=device_type)
        each_node_work_items_out = torch.ones((nodes_num, item_num[0], item_num[1], 2), device=device_type)
        each_node_work_items_len = torch.ones((nodes_num, item_num[0], item_num[1], 2), device=device_type)
        each_node_work_items_in = torch.ones((nodes_num, item_num[0], item_num[1], 2), device=device_type)
        decov_node_work_items_in_after_padding = torch.ones(
            (nodes_num, item_num[0], item_num[1], 2), device=device_type)
        # last node output.no overlap
        each_item_out_size = torch.div(fused_out, v_item_num, rounding_mode='trunc')
        if each_item_out_size[0].item() <= 1 or each_item_out_size[1].item() <= 1:
            OPT_INFO('Split block is too many')
        remain = fused_out - each_item_out_size * v_item_num
        last_item_out_len = each_item_out_size + remain
        work_items_out = work_items_out * each_item_out_size

        work_items_out[:, item_num[1] - 1, 1] = last_item_out_len[1]
        work_items_out[item_num[0] - 1, :, 0] = last_item_out_len[0]
        all_work_items_in = work_items_out[0, 0, :]
        overlap = torch.zeros((item_num[0], item_num[1], 2), dtype=torch.int16, device=device_type)
        for idx in range(end_split_node_idx, start_split_node_idx - 1, -1):
            node = graph.nodes[idx]
            if 'kernel_x' in node.params:
                dl_x = node.get_param('dilation_x', optional=True, default_value=1)
                dl_y = node.get_param('dilation_y', optional=True, default_value=1)
                kernel_size = torch.tensor([(node.params['kernel_y'] - 1) * dl_y + 1,
                                            (node.params['kernel_x'] - 1) * dl_x + 1], device=device_type)
                padding, stride = fill_padding_param_for_all_items(node)
                each_node_work_items_out[idx - start_split_node_idx] = work_items_out

                work_items_len = calc_work_item_size_by_output(work_items_out, kernel_size, stride, padding)
                # pooling op, out = (hw-k+padding)/stride + 1
                # if (hw-k+padding)%stride !=0, ceil_mode=true,need adjust
                if 'ceil_mode' in node.params and node.params['ceil_mode']:
                    error = pooling_ceil_mode_padding(node)
                    work_items_len[:, item_num[1] - 1, 1] = work_items_len[:, item_num[1] - 1, 1] - error[1]
                    work_items_len[item_num[0] - 1, :, 0] = work_items_len[item_num[0] - 1, :, 0] - error[0]
                overlap = calc_work_item_overlap(work_items_len, all_work_items_in, stride, padding)

                ######deconv case####################################
                # 1. transfer deconv to conv, w =(w-1)*stride+1, newpad = k-padding-1, stride=1
                # 2. calc work_items_len and overlap with conv layer's method
                # 3. update conv paramter to deconv param
                if node.type == OpType.ConvTranspose:
                    decov_node_work_items_in_after_padding[idx - start_split_node_idx, :] = (work_items_len - overlap)
                    stride = torch.tensor([node.params['stride_y'], node.params['stride_x']], device=device_type)
                    temp_len = torch.zeros_like(work_items_len)
                    left_top = torch.tensor([0, 0], device=device_type)
                    for item_y in range(item_num[0]):
                        left_top[1] = 0
                        for item_x in range(item_num[1]):
                            paded_hw = work_items_len[item_y, item_x, :]
                            hw = (paded_hw - left_top) // stride
                            compensate_point = (paded_hw - left_top) % stride != 0
                            hw = hw + compensate_point
                            remain[:] = (kernel_size[:] - (paded_hw - left_top[:]) % stride[:]) % stride[:]
                            left_top[1] = remain[1] - 1 + stride[1] * (remain[1] == 0)
                            overlap[item_y, item_x, 1] = (overlap[item_y, item_x, 1] - left_top[1]) // stride[1] + \
                                ((overlap[item_y, item_x, 1] - left_top[1]) % stride[1] != 0)
                            temp_len[item_y, item_x, :] = hw
                        remain[0] = (kernel_size[0] - (work_items_len[item_y, 0, 0] - left_top[0]) %
                                     stride[0]) % stride[0]
                        left_top[0] = remain[0] - 1 + stride[0] * (remain[0] == 0)
                        overlap[item_y, :, 0] = (overlap[item_y, :, 0] - left_top[0]) // stride[0] + \
                            ((overlap[item_y, :, 0] - left_top[0]) % stride[0] != 0)
                    work_items_len[:] = temp_len[:]

            elif node.type == OpType.DepthToSpace:
                work_items_len[:] = work_items_out[:]
                block_size_x = node.params['block_size_x']
                # block_size_y = graph.get_param('block_size_y')
                each_node_work_items_out[idx - start_split_node_idx, :] = work_items_len * block_size_x
            elif node.type in tranparent_op or node.type == OpType.Eltwise:
                work_items_len[:] = work_items_out[:]

                each_node_work_items_out[idx - start_split_node_idx, :] = work_items_len
            elif node.type == OpType.FullyConnected:
                work_items_len[:] = work_items_out[:]
                inw = node.inputs[0].shape[-1]
                work_items_len[..., -1] = inw
                each_node_work_items_out[idx - start_split_node_idx, :] = work_items_out
            elif node.type == OpType.MatMul:
                work_items_len[:] = work_items_out[:]
                each_node_work_items_out[idx - start_split_node_idx, :] = work_items_out
            all_work_items_in = (work_items_len - overlap)
            work_items_out[:] = work_items_len[:]
            each_node_work_items_len[idx - start_split_node_idx, :] = work_items_len
            each_node_work_items_in[idx - start_split_node_idx, :] = all_work_items_in
        return each_node_work_items_len, each_node_work_items_in, each_node_work_items_out, decov_node_work_items_in_after_padding

    ###############################################################################################

    def disp_be_able_partion_node(end_split_idx_list, start_split_idx_list):
        for i, idx in enumerate(start_split_idx_list):
            OPT_INFO(f"layer from {idx} to {end_split_idx_list[i]} are splitting.")
            for m in range(end_split_idx_list[i] - idx + 1):
                node = graph.nodes[idx + m]
                OPT_INFO(f"\tlayer id: {idx + m} (node name = {node.name}) can be split")

    ###################
    # get each separatable segment

    def get_partition_segments(item_num, ratio):
        node_idx = 0
        end_idx = len(graph.nodes)
        # node_idx = 13
        start_split_idx_list = []
        end_split_idx_list = []
        end_eltwise_idx_list = []
        search_start_idx_flag = True

        # search all separatable layer,record start_id and end_id
        # if a conv layer's output as next several layer input, should merge this conv
        def node_child_is_eltwise(n):
            if len(graph.nodes[n].children) > 0 and graph.nodes[n].children[0].type == OpType.Eltwise:
                return int(graph.nodes[n].children[0].attrs['layer_id'])
            else:
                return 0

        def check_valid_split_size(n):
            valid = False
            if 'kernel_x' in n.params:
                kernel_x = n.params['kernel_x']
                kernel_y = n.params['kernel_y']
                pad_left = n.params['pad_left']
                pad_top = n.params['pad_top']
                stride_x = n.params['stride_x']

                stride_y = n.params['stride_y']

                i_h = n.inputs[0].ir_shape[1] // item_num[0]
                i_w = n.inputs[0].ir_shape[2] // item_num[1]
                o_h = n.outputs[0].ir_shape[1] // item_num[0]
                o_w = n.outputs[0].ir_shape[2] // item_num[1]
                valid = (1-kernel_y+pad_top <= (o_h-1)*stride_y) and (1-kernel_x+pad_left+1 <= (o_w-1)*stride_x)
            return valid

        def save_split_end_idx(node_idx):
            end_split_idx_list.append(node_idx)
            end_eltwise_idx_list.append(node_child_is_eltwise(node_idx))

        if partition_based_on_sram_size:
            item_num = [2, 2]
        while node_idx < end_idx:
            n = graph.nodes[node_idx]
            # check if separatable layer
            # partitioning decision
            not_slidewin_op_split = False
            k_in_range_of_feature = False
            if 'kernel_x' in n.params:
                if n.type != OpType.ConvTranspose:
                    if partition_based_on_sram_size:
                        k_in_range_of_feature = n.inputs[0].ir_shape[1] // item_num[0] > n.params['kernel_y'] \
                            and (n.outputs[0].ir_shape[1] // item_num[0] > 0) \
                            and n.inputs[0].ir_shape[2] // item_num[1] > n.params['kernel_x'] \
                            and ((n.inputs[0].ir_shape[1] * n.inputs[0].ir_shape[2] *
                                  n.inputs[0].ir_shape[3] >= sram_size * 512)
                                 or (n.outputs[0].ir_shape[1] * n.outputs[0].ir_shape[2] *
                                     n.outputs[0].ir_shape[3] >= sram_size * 512))
                    else:
                        k_in_range_of_feature = check_valid_split_size(n)
                else:
                    k_in_range_of_feature = (n.inputs[0].ir_shape[1] > n.params['kernel_y'])
            elif (not search_start_idx_flag) and n.type in tranparent_op and 'kernel_x' in n.parents[0].params:
                if n.inputs[0].shape == n.outputs[0].shape or n.type == OpType.Transpose:
                    k_in_range_of_feature = True
            elif n.type == OpType.DepthToSpace:
                not_slidewin_op_split = True
            # fc only support h dir split
            elif (n.type == OpType.FullyConnected and item_num[0] > 1) or n.type == OpType.MatMul:
                not_slidewin_op_split = (n.inputs[0].ir_shape[-2] > item_num[0]
                                         ) and (n.outputs[0].ir_shape[-1] > item_num[1])
            else:
                k_in_range_of_feature = False
            if k_in_range_of_feature:
                if search_start_idx_flag:
                    start_split_idx_list.append(node_idx)
                    search_start_idx_flag = False
                    continue
                # check overlap , if overlap is too large, stop here
                w_len, w_in, _, _ = get_partition_params_by_back_propagate(node_idx, start_split_idx_list[-1])
                overlap_rate = 1 - w_in[0, 0, 0, :] / w_len[0, 0, 0, :]
                if overlap_rate[0].item() > ratio or overlap_rate[1].item() > ratio:
                    if len(end_split_idx_list) == 0 or node_idx not in start_split_idx_list:
                        node_idx = node_idx - 1
                    search_start_idx_flag = True
                    # end_split_idx_list.append(node_idx)
                    save_split_end_idx(node_idx)
                    # if overlap is out_of_range and only one op, it indicates this op can't be split
                    if start_split_idx_list[-1] == node_idx:
                        del start_split_idx_list[-1]
                        del end_split_idx_list[-1]
                        del end_eltwise_idx_list[-1]

                elif (node_idx + 1) != end_idx:

                    # check conv's output name, it maybe next several layers' input
                    # if it is several-conv's layer input, seperated operation should stop here
                    if len(n.children) == 0 or (n.type == OpType.Eltwise and check_valid_split_size(n.parents[0])) or \
                            (n.children[0].type not in tranparent_op and 'kernel_x' not in n.children[0].params):
                        search_start_idx_flag = True
                        # end_split_idx_list.append(node_idx)
                        save_split_end_idx(node_idx)
                    elif n.type == OpType.Eltwise and 'kernel_x' in n.parents[1].params:
                        search_start_idx_flag = True
                        # end_split_idx_list.append(node_idx-1)
                        save_split_end_idx(node_idx - 1)
                    elif len(n.children) > 1:
                        search_start_idx_flag = True
                        # end_split_idx_list.append(node_idx)
                        save_split_end_idx(node_idx)
                    elif len(n.children[0].inputs) > 1:
                        # check child node if eltwise is or not
                        if n.children[0].parents[1].name != n.name:
                            search_start_idx_flag = True
                            # end_split_idx_list.append(node_idx)
                            save_split_end_idx(node_idx)

            else:
                if (not search_start_idx_flag):
                    search_start_idx_flag = True
                    # end_split_idx_list.append(node_idx-1)
                    save_split_end_idx(node_idx - 1)

            # #split op alone
            if not_slidewin_op_split:
                start_split_idx_list.append(node_idx)
                # end_split_idx_list.append(node_idx)
                save_split_end_idx(node_idx)
                search_start_idx_flag = True
            node_idx += 1
        if not search_start_idx_flag:
            # end_split_idx_list.append(node_idx-1)
            save_split_end_idx(node_idx - 1)
        # re-check if eltwise op can be merged
        for i in range(len(end_eltwise_idx_list)):
            if end_eltwise_idx_list[i] > 0:
                idx = np.where(end_eltwise_idx_list[i] == np.array(end_eltwise_idx_list))

                if idx[0].size == 1:
                    # the other input is not conv, so ignore
                    # if graph.nodes[end_eltwise_idx_list[i]].parents[0].type != OpType.Transpose \
                    #         and graph.nodes[end_eltwise_idx_list[i]].parents[1].type != OpType.Transpose:
                    #     end_split_idx_list[idx[0][0]] = end_eltwise_idx_list[i]
                    #     end_eltwise_idx_list[i] = end_eltwise_idx_list[i] + tot_nodes_num
                    # else:
                    end_eltwise_idx_list[i] = 0
                else:
                    end_split_idx_list[idx[0][1]] = end_eltwise_idx_list[i]
        disp_be_able_partion_node(end_split_idx_list, start_split_idx_list)
        return end_split_idx_list, start_split_idx_list, end_eltwise_idx_list

    def copy_quant_paramters(new_outp_tensor, replaced_node):
        new_outp_tensor.qmin = replaced_node.outputs[0].qmin
        new_outp_tensor.qmax = replaced_node.outputs[0].qmax
        new_outp_tensor.similarity = replaced_node.outputs[0].similarity
        new_outp_tensor.scale = replaced_node.outputs[0].scale
        new_outp_tensor.zerop = replaced_node.outputs[0].zerop
        new_outp_tensor.qbits = replaced_node.outputs[0].qbits
        new_outp_tensor.ir_dtype = replaced_node.outputs[0].ir_dtype
        new_outp_tensor.dtype = replaced_node.outputs[0].dtype
        new_outp_tensor.qinvariant = replaced_node.outputs[0].qinvariant
        new_outp_tensor.key_axis = replaced_node.outputs[0].key_axis

    #################################################################################################
    ############################insert slice op######################################################
    # fc case,x direction shape is at constants(weights,so slice item_x is 1)

    def insert_slice_op(all_work_items_len, all_work_items_in, insert_node_pos, inserted_op_list):

        parts_num_y = item_num[0]
        parts_num_x = item_num[1]
        if graph.nodes[insert_node_pos].type == OpType.FullyConnected:
            parts_num_x = 1
        # inserted_op_list = []
        crop_begin_pos = [0, 0]
        crop_end_pos = [0, 0]
        for item_y in range(parts_num_y):
            crop_begin_pos[1] = 0
            for item_x in range(parts_num_x):
                current_node = graph.nodes[insert_node_pos]
                parent = current_node.parents[0]  # graph.nodes[insert_node_pos-1]

                slice_name = current_node.name + '_slice_item_y' + str(item_y) + '_item_x' + str(item_x)
                slice = PyNode(graph.get_valid_node_name(slice_name), OpType.Slice)

                shape_dim = len(current_node.inputs[0].shape)
                # fill input and outputs tensor
                # add inputs from curren inserted node
                # slice.add_input(current_node.inputs[0])
                slice.add_input(parent.outputs[0])
                # if len(slice.parents) == 0:
                #     slice.add_parent(parent)
                # new slice output
                slice_outp_tensor_name = graph.get_valid_tensor_name(
                    current_node.inputs[0].name + '_slice' + '_item_y' + str(item_y) + '_item_x' + str(item_x))
                # h,w size
                h = all_work_items_len[0, item_y, item_x, 0].int().item()
                w = all_work_items_len[0, item_y, item_x, 1].int().item()
                if shape_dim == 4:
                    outshape = TensorShape([current_node.inputs[0].shape[0], h, w, current_node.inputs[0].shape[3]])
                else:
                    outshape = TensorShape([h, w])
                out_tensor = PyTensor(slice_outp_tensor_name, outshape, current_node.inputs[0].dtype)
                out_tensor.ir_shape = outshape
                out_tensor.qmin = current_node.inputs[0].qmin
                out_tensor.qmax = current_node.inputs[0].qmax
                out_tensor.ir_dtype = current_node.inputs[0].ir_dtype
                slice.add_output(out_tensor)
                # need setting slice paramsters:('strides'),('begin'),('end')
                slice.params['strides'] = [1] * shape_dim
                if shape_dim == 4:
                    slice.params['begin'] = [0, crop_begin_pos[0], crop_begin_pos[1], 0]

                    crop_end_pos[0] = crop_begin_pos[0] + all_work_items_len[0, item_y, item_x, 0].int().item()
                    crop_end_pos[1] = crop_begin_pos[1] + all_work_items_len[0, item_y, item_x, 1].int().item()
                    slice.params['end'] = [-1, crop_end_pos[0], crop_end_pos[1], out_tensor.ir_shape[3]]
                else:
                    slice.params['begin'] = [crop_begin_pos[0], crop_begin_pos[1]]

                    crop_end_pos[0] = crop_begin_pos[0] + all_work_items_len[0, item_y, item_x, 0].int().item()
                    crop_end_pos[1] = crop_begin_pos[1] + all_work_items_len[0, item_y, item_x, 1].int().item()
                    slice.params['end'] = [crop_end_pos[0], crop_end_pos[1]]
                slice.params['upper_bound'] = True
                slice.attrs.update(parent.attrs.clone())
                # x direction offset update
                crop_begin_pos[1] = crop_begin_pos[1] + all_work_items_in[0, item_y, item_x, 1].int().item()
                inserted_op_list.append(slice)
            # y direction offset update
            crop_begin_pos[0] = crop_begin_pos[0] + all_work_items_in[0, item_y, 0, 0].int().item()

    def insert_crop_op(all_work_items_len, all_work_items_in, insert_node_pos, inserted_op_list, parents_id=0):

        parts_num_y = item_num[0]
        parts_num_x = item_num[1]
        if graph.nodes[insert_node_pos].type == OpType.FullyConnected:
            parts_num_x = 1
        # inserted_op_list = []
        crop_begin_pos = [0, 0]
        crop_end_pos = [0, 0]
        do_concat_subblock_crop = False
        current_node = graph.nodes[insert_node_pos]
        if sram_size > 0 and concat_block > 1 and len(current_node.parents) > parents_id + 1:
            do_concat_subblock_crop = True
            # remove residual inputs
            for i in range(parents_id + 1):
                current_node.remove_input(current_node.inputs[0])
            concat_subblock_h = current_node.inputs[0].shape[1] // concat_block
        for item_y in range(parts_num_y):
            crop_begin_pos[1] = 0
            for item_x in range(parts_num_x):
                # current_node = graph.nodes[insert_node_pos]
                if do_concat_subblock_crop:
                    # currently, only suppport h direction split
                    concat_subblock_id = crop_begin_pos[0] // concat_subblock_h
                    concat_subblock_id = concat_subblock_id - int(concat_subblock_id == parts_num_y - 1)
                    parent = current_node.parents[concat_subblock_id]  # graph.nodes[insert_node_pos-1]

                    crop_name = current_node.name + '_crop_item_y' + \
                        str(item_y) + 'node_pos' + str(len(graph.nodes))+timestamp_string()
                    crop = PyNode(graph.get_valid_node_name(crop_name), OpType.Crop)

                    # fill input and outputs tensor
                    # add inputs from curren inserted node
                    crop.add_input(parent.outputs[0])

                    crop_outp_tensor_name = graph.get_valid_tensor_name(
                        current_node.inputs[concat_subblock_id].name + '_crop' + '_item_y' + str(item_y) + '_node_pos' +
                        str(len(inserted_op_list) + len(graph.nodes))+timestamp_string())
                    # h,w size
                    h = all_work_items_len[0, item_y, item_x, 0].int().item()
                    w = all_work_items_len[0, item_y, item_x, 1].int().item()
                    # only shape_dim == 4:
                    outshape = TensorShape([current_node.inputs[0].shape[0], h, w, current_node.inputs[0].shape[3]])

                    out_tensor = PyTensor(crop_outp_tensor_name, outshape, current_node.inputs[0].dtype)
                    out_tensor.ir_shape = outshape
                    out_tensor.qmin = current_node.inputs[0].qmin
                    out_tensor.qmax = current_node.inputs[0].qmax
                    out_tensor.ir_dtype = current_node.inputs[0].ir_dtype
                    crop.add_output(out_tensor)
                    # begin=[0,0,0,0]
                    # end=[-1,165,272,64]
                    crops = [[0, 0], [0, 0], [0, 0], [0, 0]]
                    # need setting crops paramsters:[('begin'),('end')]
                    # only support 4 dim case
                    # h dir
                    # crop_end_pos[0] = crop_begin_pos[0]+all_work_items_len[0, item_y, item_x, 0].int().item()
                    # w dir
                    crop_end_pos[1] = crop_begin_pos[1] + all_work_items_len[0, item_y, item_x, 1].int().item()
                    # batch dim end
                    crops[0][1] = out_tensor.ir_shape[0]
                    # h dim start
                    crops[1][0] = crop_begin_pos[0] - concat_subblock_id * concat_subblock_h
                    # h dim end
                    crops[1][1] = crops[1][0] + all_work_items_len[0, item_y, item_x, 0].int().item()
                    # w dim start
                    crops[2][0] = crop_begin_pos[1]
                    # w dim end
                    crops[2][1] = crop_end_pos[1]
                    # c dim end
                    crops[3][1] = out_tensor.ir_shape[3]
                    crop.params['crops'] = crops
                    crop.attrs.update(parent.attrs.clone())

                else:
                    parent = current_node.parents[parents_id]  # graph.nodes[insert_node_pos-1]

                    crop_name = current_node.name + '_crop_item_y' + \
                        str(item_y) + '_item_x' + str(item_x) + timestamp_string()
                    crop = PyNode(graph.get_valid_node_name(crop_name), OpType.Crop)

                    shape_dim = len(current_node.inputs[0].shape)
                    # fill input and outputs tensor
                    # add inputs from curren inserted node
                    # slice.add_input(current_node.inputs[0])
                    crop.add_input(parent.outputs[0])
                    # if len(slice.parents) == 0:
                    #     slice.add_parent(parent)
                    # new slice output
                    crop_outp_tensor_name = graph.get_valid_tensor_name(
                        current_node.inputs[0].name + '_crop' + '_item_y' + str(item_y) + '_item_x' + str(item_x)+timestamp_string())
                    # h,w size
                    h = all_work_items_len[0, item_y, item_x, 0].int().item()
                    w = all_work_items_len[0, item_y, item_x, 1].int().item()
                    if shape_dim == 4:
                        outshape = TensorShape([current_node.inputs[0].shape[0], h,
                                                w, current_node.inputs[0].shape[3]])
                    else:
                        outshape = TensorShape([h, w])
                    out_tensor = PyTensor(crop_outp_tensor_name, outshape, current_node.inputs[0].dtype)
                    out_tensor.ir_shape = outshape
                    out_tensor.qmin = current_node.inputs[0].qmin
                    out_tensor.qmax = current_node.inputs[0].qmax
                    out_tensor.ir_dtype = current_node.inputs[0].ir_dtype
                    crop.add_output(out_tensor)
                    crop.attrs.update(parent.attrs.clone())
                    # begin=[0,0,0,0]
                    # end=[-1,165,272,64]
                    crops = [[0, 0], [0, 0], [0, 0], [0, 0]]
                    # need setting crops paramsters:[('begin'),('end')]
                    if shape_dim == 4:
                        crop_end_pos[0] = crop_begin_pos[0] + all_work_items_len[0, item_y, item_x, 0].int().item()
                        crop_end_pos[1] = crop_begin_pos[1] + all_work_items_len[0, item_y, item_x, 1].int().item()
                        crops[0][1] = out_tensor.ir_shape[0]
                        crops[1][0] = crop_begin_pos[0]
                        crops[1][1] = crop_end_pos[0]
                        crops[2][0] = crop_begin_pos[1]
                        crops[2][1] = crop_end_pos[1]

                        crops[3][1] = out_tensor.ir_shape[3]
                        crop.params['crops'] = crops

                    else:
                        crops = [[0, 0], [0, 0]]
                        crop_end_pos[0] = crop_begin_pos[0] + all_work_items_len[0, item_y, item_x, 0].int().item()
                        crop_end_pos[1] = crop_begin_pos[1] + all_work_items_len[0, item_y, item_x, 1].int().item()
                        crops[0][0] = crop_begin_pos[0]
                        crops[0][1] = crop_end_pos[0]
                        crops[1][0] = crop_begin_pos[1]
                        crops[1][1] = crop_end_pos[1]
                        crop.attrs['batch_size_in_IR'] = 0
                        crop.params['crops'] = crops
                # x direction offset update
                crop_begin_pos[1] = crop_begin_pos[1] + all_work_items_in[0, item_y, item_x, 1].int().item()
                inserted_op_list.append(crop)
            # y direction offset update
            crop_begin_pos[0] = crop_begin_pos[0] + all_work_items_in[0, item_y, 0, 0].int().item()

    ############insert matmul two inputs slice###########################################
    # matmul partition number = item_num[0]+item_num[1]
    def insert_matmul_slice_op(all_work_items_len, all_work_items_in, insert_node_pos, inserted_op_list):
        # inserted_op_list = []
        crop_begin_pos = [0, 0]
        crop_end_pos = [0, 0]
        crop_begin_pos[1] = 0
        current_node = graph.nodes[insert_node_pos]

        for i in range(1, -1, -1):
            parent = current_node.parents[i]
            if i == 0:  # h dir
                parts_num_y = item_num[i]
                parts_num_x = 1
                crop_end_pos[1] = current_node.inputs[0].shape[-1]
                crop_begin_pos[0] = 0
            else:  # w dir
                parts_num_x = item_num[i]
                parts_num_y = 1
                crop_end_pos[0] = current_node.inputs[1].shape[-2]
                crop_begin_pos[1] = 0
            for item_y in range(parts_num_y):
                for item_x in range(parts_num_x):
                    slice_name = current_node.name + '_slice_item_y' + str(item_y + parts_num_x) + '_item_x' + str(
                        item_x)
                    slice = PyNode(graph.get_valid_node_name(slice_name), OpType.Slice)
                    shape_dim = len(current_node.inputs[i].shape)
                    # fill input and outputs tensor
                    # new slice output
                    slice_outp_tensor_name = graph.get_valid_tensor_name(
                        current_node.inputs[i].name + '_slice' + '_item_y' + str(item_y) + '_item_x' + str(item_x))
                    # h,w size
                    h_w = all_work_items_len[0, item_y, item_x, i].int().item()
                    inshape = (current_node.inputs[i].shape)
                    if i == 1:
                        outshape = (inshape[0], inshape[1], inshape[2], h_w)
                    else:
                        outshape = (inshape[0], inshape[1], h_w, inshape[3])
                    out_tensor = PyTensor(slice_outp_tensor_name, outshape, current_node.inputs[i].dtype)
                    out_tensor.ir_shape = outshape
                    out_tensor.qmin = current_node.inputs[i].qmin
                    out_tensor.qmax = current_node.inputs[i].qmax
                    out_tensor.ir_dtype = current_node.inputs[i].ir_dtype
                    slice.add_output(out_tensor)

                    # add inputs from curren inserted node
                    slice.add_input(parent.outputs[0])
                    # if len(slice.parents) == 0:
                    #     slice.add_parent(parent)
                    # slice.add_input(current_node.inputs[i])
                    # need setting slice paramsters:('strides'),('begin'),('end')
                    slice.params['strides'] = [1] * shape_dim
                    begin = [0, 0, 0, 0]
                    begin[-2 + i] = crop_begin_pos[i]
                    slice.params['begin'] = begin

                    crop_end_pos[i] = crop_begin_pos[i] + all_work_items_len[0, item_y, item_x, i].int().item()
                    slice.params['end'] = [-1, -1, crop_end_pos[0], crop_end_pos[1]]

                    slice.params['upper_bound'] = True
                    slice.attrs.update(parent.attrs.clone())
                    # x direction offset update
                    crop_begin_pos[i] = crop_begin_pos[i] + all_work_items_in[0, item_y, item_x, i].int().item()
                    inserted_op_list.append(slice)
            # parent.remove_child(parent.children[0])

    #############################################################################################################
    ############################insert slide window type op######################################################

    def insert_slide_window_op(inputs, each_node_work_items_out, all_work_items_len, dedecov_node_work_items_in_after_paddingconv, first_node_idx, node,
                               inserted_op_list, item_x, item_y):

        optype = graph.nodes[node].type
        replaced_node = graph.nodes[node]
        new_layer_name = replaced_node.name
        part = PyNode(graph.get_valid_node_name(new_layer_name + '_item_y' +
                                                str(item_y) + '_item_x' + str(item_x) + '_node' + str(node)), optype)
        # add input tensor
        part.add_input(inputs)
        # add output tensor
        h = each_node_work_items_out[node - first_node_idx, item_y, item_x, 0].int().item()
        w = each_node_work_items_out[node - first_node_idx, item_y, item_x, 1].int().item()
        new_outp_tensor_name = graph.get_valid_tensor_name(
            replaced_node.outputs[0].name + '_item_y' + str(item_y) + '_item_x' + str(item_x) + '_node' + str(node))
        newshape = TensorShape([replaced_node.outputs[0].shape[0], h, w, replaced_node.outputs[0].shape[3]])
        new_outp_tensor = PyTensor(new_outp_tensor_name, newshape, replaced_node.outputs[0].dtype)
        new_outp_tensor.ir_shape = newshape

        copy_quant_paramters(new_outp_tensor, replaced_node)
        part.add_output(new_outp_tensor)
        inputs = part.outputs[0]

        # add conv paramers
        for k, v in replaced_node.params.items():
            part.params[k] = copy.deepcopy(v)
        part.attrs.update(replaced_node.attrs.clone())
        for k, v in replaced_node.constants.items():
            part.constants[k] = replaced_node.constants[k]
        # padding need update
        if replaced_node.type == OpType.ConvTranspose:
            padding = fill_deconv_padding_param_for_all_items(
                replaced_node, node - first_node_idx, all_work_items_len, each_node_work_items_out, dedecov_node_work_items_in_after_paddingconv)
        else:
            padding, _ = fill_padding_param_for_all_items(replaced_node)
        part.params['pad_top'] = padding[item_y, item_x, 0].item()
        part.params['pad_left'] = padding[item_y, item_x, 1].item()
        part.params['pad_bottom'] = padding[item_y, item_x, 2].item()
        part.params['pad_right'] = padding[item_y, item_x, 3].item()

        inserted_op_list.append(part)
        return inputs

    #############################################################################################################
    ############################insert not slide window type op##################################################

    def insert_non_slide_window_op(inputs, each_node_work_items_out, all_work_items_in, first_node_idx, node,
                                   inserted_op_list, item_x, item_y):

        optype = graph.nodes[node].type
        replaced_node = graph.nodes[node]
        new_layer_name = replaced_node.name
        part = PyNode(graph.get_valid_node_name(new_layer_name + '_item_y' +
                                                str(item_y) + '_item_x' + str(item_x) + '_node' + str(node)), optype)
        # add input tensor
        part.add_input(inputs)
        # add output tensor
        shape_dim = len(replaced_node.outputs[0].shape)
        h = each_node_work_items_out[node - first_node_idx, item_y, item_x, 0].int().item()
        w = each_node_work_items_out[node - first_node_idx, item_y, item_x, 1].int().item()
        new_outp_tensor_name = graph.get_valid_tensor_name(
            replaced_node.outputs[0].name + '_item_y' + str(item_y) + '_item_x' + str(item_x) + '_node' + str(node))
        if replaced_node.type == OpType.Transpose:
            newshape = TensorShape([replaced_node.outputs[0].shape[0], replaced_node.outputs[0].shape[1], h, w])
        elif shape_dim == 4:
            newshape = TensorShape([replaced_node.outputs[0].shape[0], h, w, replaced_node.outputs[0].shape[3]])
        else:
            newshape = TensorShape([h, w])
        new_outp_tensor = PyTensor(new_outp_tensor_name, newshape, replaced_node.outputs[0].dtype)
        new_outp_tensor.ir_shape = newshape
        copy_quant_paramters(new_outp_tensor, replaced_node)
        part.add_output(new_outp_tensor)
        inputs = part.outputs[0]

        # add not conv type paramers
        for k, v in replaced_node.params.items():
            part.params[k] = copy.deepcopy(v)
        part.attrs.update(replaced_node.attrs.clone())
        # replaced_node.params['method'] == 'PRELU' need split constants
        # partition constants negative_slope
        if 'method' in replaced_node.params and replaced_node.params['method'] == 'PRELU':

            next_h_pos = 0
            next_w_pos = 0
            for y in range(1, item_y+1, 1):
                next_h_pos = next_h_pos + all_work_items_in[node - first_node_idx, y-1, item_x, 0].int().item()
            for x in range(1, item_x+1, 1):
                next_w_pos = next_w_pos + all_work_items_in[node - first_node_idx, item_y, x-1, 1].int().item()

            for k, v in replaced_node.constants.items():
                irshape = replaced_node.constants[k].ir_shape
                newshape = (irshape[0], h, w, irshape[3])

                part.constants[k] = copy.deepcopy(replaced_node.constants[k])
                if replaced_node.constants[k].betensor.ndim == 3:
                    part.constants[k].betensor = copy.deepcopy(
                        replaced_node.constants[k].betensor[next_h_pos:next_h_pos + h, next_w_pos:next_w_pos + w, ...])
                else:
                    part.constants[k].betensor = copy.deepcopy(
                        replaced_node.constants[k].betensor[:, next_h_pos:next_h_pos + h, next_w_pos:next_w_pos + w,
                                                            ...])
                part.constants[k].betensor = part.constants[k].betensor.reshape(newshape)
        else:
            for k, v in replaced_node.constants.items():
                part.constants[k] = replaced_node.constants[k]

        inserted_op_list.append(part)
        return inputs

    #############################################################################################################
    ############################insert fullyconnected op##################################################
    def insert_fc_op(inputs, each_node_work_items_out, first_node_idx, cur_node_idx, inserted_op_list, item_y,
                     parts_num_x):

        optype = graph.nodes[cur_node_idx].type
        replaced_node = graph.nodes[cur_node_idx]
        new_layer_name = replaced_node.name
        last_w = 0
        for item_x in range(parts_num_x):
            part = PyNode(graph.get_valid_node_name(new_layer_name + '_item_y' + str(item_y) +
                                                    '_item_x' + str(item_x) + '_node' + str(cur_node_idx)), optype)
            # add input tensor
            part.add_input(inputs)
            # add output tensor
            h = each_node_work_items_out[cur_node_idx - first_node_idx, item_y, item_x, 0].int().item()
            w = each_node_work_items_out[cur_node_idx - first_node_idx, item_y, item_x, 1].int().item()
            new_outp_tensor_name = graph.get_valid_tensor_name(
                replaced_node.outputs[0].name + '_item_y' + str(item_y) + '_item_x' + str(item_x) + '_node' + str(
                    cur_node_idx))

            newshape = TensorShape([h, w])
            new_outp_tensor = PyTensor(new_outp_tensor_name, newshape, replaced_node.outputs[0].dtype)
            new_outp_tensor.ir_shape = newshape
            copy_quant_paramters(new_outp_tensor, replaced_node)
            part.add_output(new_outp_tensor)
            # inputs = part.outputs[0]
            # add conv paramers
            for k, v in replaced_node.params.items():
                part.params[k] = copy.deepcopy(v)
            part.attrs.update(replaced_node.attrs.clone())
            # partition constants weight
            for k, v in replaced_node.constants.items():
                newshape = (w,)
                for i in range(len(replaced_node.constants[k].ir_shape)-1):
                    newshape = newshape + (replaced_node.constants[k].ir_shape[i+1],)
                # tmp = PyTensor(replaced_node.constants[k].name, newshape, replaced_node.constants[k].dtype)
                # tmp=(replaced_node.constants[k])  # tmp copy from replaced_node.constants[k],return old tmp
                part.constants[k] = copy.deepcopy(replaced_node.constants[k])
                part.constants[k].betensor = copy.deepcopy(
                    replaced_node.constants[k].betensor[last_w:last_w + w, ...])
                part.constants[k].ir_shape = newshape
            last_w = last_w + w
            inserted_op_list.append(part)
        return inputs

    def insert_matmul_op(inputs, each_node_work_items_out, first_node_idx, cur_node_idx, inserted_op_list, item_y,
                         item_x):

        optype = graph.nodes[cur_node_idx].type
        replaced_node = graph.nodes[cur_node_idx]
        new_layer_name = replaced_node.name

        part = PyNode(graph.get_valid_node_name(new_layer_name + '_item_y' + str(item_y) +
                                                '_item_x' + str(item_x) + '_node' + str(cur_node_idx)), optype)
        # add input tensor
        part.add_input(inputs[0])
        part.add_input(inputs[1])
        # add output tensor

        h = each_node_work_items_out[cur_node_idx - first_node_idx, item_y, item_x, 0].int().item()
        w = each_node_work_items_out[cur_node_idx - first_node_idx, item_y, item_x, 1].int().item()
        new_outp_tensor_name = graph.get_valid_tensor_name(
            replaced_node.outputs[0].name + '_item_y' + str(item_y) + '_item_x' + str(item_x) + '_node' + str(
                cur_node_idx))

        outshape = (replaced_node.outputs[0].shape)
        newshape = (outshape[0], outshape[1], h, w)

        new_outp_tensor = PyTensor(new_outp_tensor_name, newshape, replaced_node.outputs[0].dtype)
        new_outp_tensor.ir_shape = newshape
        copy_quant_paramters(new_outp_tensor, replaced_node)
        part.add_output(new_outp_tensor)
        # add  paramers
        for k, v in replaced_node.params.items():
            part.params[k] = copy.deepcopy(v)
        part.attrs.update(replaced_node.attrs.clone())
        for k, v in replaced_node.constants.items():
            part.constants[k] = replaced_node.constants[k]
        inserted_op_list.append(part)
        return inputs

    def insert_eltwise_op(inputs, each_node_work_items_out, first_node_idx, cur_node_idx, inserted_op_list, item_y,
                          item_x):

        optype = graph.nodes[cur_node_idx].type
        replaced_node = graph.nodes[cur_node_idx]
        new_layer_name = replaced_node.name
        shape_dim = len(replaced_node.outputs[0].shape)
        part = PyNode(graph.get_valid_node_name(new_layer_name + '_item_y' + str(item_y) +
                                                '_item_x' + str(item_x) + '_node' + str(cur_node_idx)), optype)
        # add input tensor
        part.add_input(inputs[0])
        part.add_input(inputs[1])
        # add output tensor

        h = each_node_work_items_out[cur_node_idx - first_node_idx, item_y, item_x, 0].int().item()
        w = each_node_work_items_out[cur_node_idx - first_node_idx, item_y, item_x, 1].int().item()
        new_outp_tensor_name = graph.get_valid_tensor_name(
            replaced_node.outputs[0].name + '_item_y' + str(item_y) + '_item_x' + str(item_x) + '_node' + str(
                cur_node_idx))

        if shape_dim == 4:
            newshape = TensorShape([replaced_node.outputs[0].shape[0], h, w, replaced_node.outputs[0].shape[3]])
        else:
            newshape = TensorShape([h, w])

        new_outp_tensor = PyTensor(new_outp_tensor_name, newshape, replaced_node.outputs[0].dtype)
        new_outp_tensor.ir_shape = newshape
        copy_quant_paramters(new_outp_tensor, replaced_node)
        part.add_output(new_outp_tensor)
        # add not conv type paramers

        for k, v in replaced_node.params.items():
            if type(v) != TensorShape:
                part.params[k] = copy.deepcopy(v)
            else:
                part.params[k] = v.clone()
        part.attrs.update(replaced_node.attrs.clone())
        for i in range(len(replaced_node.placeholders)):
            part.placeholders.append(replaced_node.placeholders[i].clone(replaced_node.placeholders[i].name))

        inserted_op_list.append(part)
        return inputs

    def get_eltwise_input0(cur_node_idx, num_x, num_y, node_num, eltwise_inputs):
        eltwise_inputs.clear()
        tot_slice = num_x * num_y
        for i in range(tot_slice):
            eltwise_inputs.append(cur_node_idx + tot_slice + (node_num - 1) + i * node_num)

    def check_concat_total_or_subblock(graph, current_node, segment_items_size, segment, start_split_idx_list,
                                       node_offset):
        cond = concat_block > 1 and sram_size > 0 \
            and len(current_node.children) >= 2 \
            and len(current_node.children) < 4
        if cond:
            segment_inc = 0
            for i in range(len(current_node.children)):
                if 'kernel_x' in current_node.children[i].params and cond:
                    segment_inc = segment_inc + 1
                    next_start_id = start_split_idx_list[segment + segment_inc] + node_offset
                    cond = cond and next_start_id in start_split_idx_list
                    cond = cond and (graph.nodes[next_start_id].name == current_node.children[i].name)
                    cond = cond and (segment_items_size[segment + segment_inc, 0] == segment_items_size[segment, 0])
        return cond

    #########################################################################################################
    # partition_based_on_sram_size is False, if seperate op for debug footprint, if cfg file has
    # featuremap_splits_sram_size and its value is big than 0, featuremap split is used to reduce footprint
    partition_based_on_sram_size = sram_size > 0
    # to reduce footprint, we need split conv op as many as possible, whatever overlap rate cfg
    if partition_based_on_sram_size:
        ratio = 0.99
    device_type = graph.nodes[0].outputs[0].betensor.device
    # get each seperatable segment
    if len(tiling_list) == 0:
        end_split_idx_list, start_split_idx_list, end_eltwise_idx_list = get_partition_segments(item_num, ratio)
    if len(tiling_list) == 0 and len(end_split_idx_list) == 0:
        return
    node_offset = 0

    #  to generate sigle core partition to debug memory footprint
    if len(tiling_list) > 0:
        start_split_idx_list = [int(tiling_list[segment * 4]) for segment in range(len(tiling_list) // 4)]
        end_split_idx_list = [int(tiling_list[segment * 4 + 1]) for segment in range(len(tiling_list) // 4)]
        end_eltwise_idx_list = [-1 for segment in range(len(tiling_list) // 4)]
    segment_items_size = torch.ones((len(end_split_idx_list), 2), device=device_type).int()
    if len(tiling_list) > 0:
        for segment in range(len(tiling_list) // 4):
            segment_items_size[segment, 0] = int(tiling_list[segment * 4 + 2])
            segment_items_size[segment, 1] = int(tiling_list[segment * 4 + 3])
    elif sram_size > 0:

        for segment in range(len(end_split_idx_list)):
            n = graph.nodes[end_split_idx_list[segment]]
            itemy = 1
            if 'kernel_x' in n.params or end_eltwise_idx_list[segment] > 0:
                weight_size = 0
                if end_split_idx_list[segment] > start_split_idx_list[segment]:
                    if end_eltwise_idx_list[segment] > 0:
                        weight_size = sram_size * 512
                    else:
                        #     weight_size = torch.prod(torch.tensor(n.constants['weights'].shape)).item()
                        if len(n.constants) != 0 and torch.prod(torch.tensor(n.constants['weights'].shape)).item() > sram_size * 1024:
                            OPT_INFO('CONV weight size is too large')
                    itemy = max((n.inputs[0].ir_shape[1] * n.inputs[0].ir_shape[2] * n.inputs[0].ir_shape[3] / (
                                sram_size * 1024 - weight_size)),
                                (n.outputs[0].ir_shape[1] * n.outputs[0].ir_shape[2] * n.outputs[0].ir_shape[3] / (
                                    sram_size * 1024 - weight_size)))
                    itemy = torch.ceil(torch.tensor(itemy, device=device_type)).int().item()
                    if itemy <= 1:
                        itemy = 2
                    # if (n.outputs[0].ir_shape[1]-itemy)//itemy>1 or (n.inputs[0].ir_shape[1]-itemy)//itemy>1:
                    #     itemy = itemy + 2
                    # to spilt much more blocks
                    while n.outputs[0].ir_shape[1] / itemy > 15:
                        itemy = itemy + 1
                    # avoid h dir size less than 5
                    while n.outputs[0].ir_shape[1] / itemy < 5:
                        itemy = itemy - 1
            elif n.type == OpType.Transpose:
                itemy = max(
                    (n.inputs[0].ir_shape[1] * n.inputs[0].ir_shape[2] * n.inputs[0].ir_shape[3] / (sram_size * 1024)),
                    (n.outputs[0].ir_shape[1] * n.outputs[0].ir_shape[2] * n.outputs[0].ir_shape[3] / (
                        sram_size * 1024)))
                itemy = torch.ceil(torch.tensor(itemy, device=device_type)).int()
                if itemy <= 1:
                    itemy = 2
            segment_items_size[segment, 0] = itemy
        # re-calculate items
        for segment in range(len(end_eltwise_idx_list)):
            if end_eltwise_idx_list[segment] > 0:
                idx = np.where(end_eltwise_idx_list[segment] == np.array(end_eltwise_idx_list))
                if idx[0].size == 2:
                    segment_items_size[segment, 0] = max(
                        [segment_items_size[idx[0][0], 0], segment_items_size[idx[0][1], 0]])

    else:
        segment_items_size[:, ] = torch.tensor(item_num, device=device_type)
    # insert partitiated op each segment
    eltwise_inputs = []
    all_nodes_eltwise = []
    for segment in range(len(end_split_idx_list)):

        item_num = segment_items_size[segment]
        parts_num_y = item_num[0].item()
        parts_num_x = item_num[1].item()
        if parts_num_y * parts_num_x == 1:
            continue
        last_node_idx = end_split_idx_list[segment] + node_offset
        first_node_idx = start_split_idx_list[segment] + node_offset
        all_work_items_len, all_work_items_in, each_node_work_items_out, dedecov_node_work_items_in_after_paddingconv = get_partition_params_by_back_propagate(
            last_node_idx, first_node_idx)

        #######################################################################################
        # insert slice op, get each block size
        #######################################################################################
        inserted_op_list = []
        eltwise_insert_pos_idx = 0
        if graph.nodes[last_node_idx].type != OpType.MatMul:
            insert_crop_op(all_work_items_len, all_work_items_in, first_node_idx, inserted_op_list)
        else:
            insert_matmul_slice_op(all_work_items_len, all_work_items_in, first_node_idx, inserted_op_list)
        ##########################################################################################
        # start add seperated conv node op, parts_num_y,parts_num_x conv op
        ##########################################################################################
        if graph.nodes[first_node_idx].type == OpType.FullyConnected:
            for item_y in range(parts_num_y):
                inputs = inserted_op_list[item_y].outputs[0]
                for node in range(first_node_idx, last_node_idx + 1, 1):
                    inputs = insert_fc_op(inputs, each_node_work_items_out, first_node_idx,
                                          node, inserted_op_list, item_y, parts_num_x)
        elif graph.nodes[first_node_idx].type == OpType.MatMul:
            for item_y in range(parts_num_y):
                for item_x in range(parts_num_x):
                    inputs = [inserted_op_list[item_y + parts_num_x].outputs[0], inserted_op_list[item_x].outputs[0]]
                    for node in range(first_node_idx, last_node_idx + 1, 1):
                        inputs = insert_matmul_op(inputs, each_node_work_items_out, first_node_idx,
                                                  node, inserted_op_list, item_y, item_x)
        else:

            if graph.nodes[last_node_idx].type == OpType.Eltwise:
                if end_eltwise_idx_list[segment] < tot_nodes_num:
                    eltwise_input = all_nodes_eltwise.pop()
                    end_eltwise_idx_list[segment] = 0
                else:  # the other side input is from crop
                    eltwise_work_items_len, eltwise_work_items_in, eltwise_node_work_items_out, _ = get_partition_params_by_back_propagate(
                        last_node_idx, last_node_idx)
                    parents_id = 'kernel_x' in graph.nodes[last_node_idx].parents[0].params or \
                        graph.nodes[last_node_idx].parents[0].type in tranparent_op
                    insert_crop_op(eltwise_work_items_len, eltwise_work_items_in,
                                   last_node_idx, inserted_op_list, parents_id)
                    eltwise_insert_pos_idx = len(inserted_op_list) - parts_num_x * parts_num_y

            for item_y in range(parts_num_y):
                for item_x in range(parts_num_x):
                    inputs = inserted_op_list[item_y * parts_num_x + item_x].outputs[0]
                    for node in range(first_node_idx, last_node_idx + 1, 1):
                        if 'kernel_x' in graph.nodes[node].params:
                            inputs = insert_slide_window_op(inputs,
                                                            each_node_work_items_out,
                                                            all_work_items_len,
                                                            dedecov_node_work_items_in_after_paddingconv,
                                                            first_node_idx,
                                                            node,
                                                            inserted_op_list,
                                                            item_x,
                                                            item_y)
                        elif graph.nodes[node].type == OpType.Eltwise:
                            # assure eltwise is last node
                            if end_eltwise_idx_list[segment] > tot_nodes_num:
                                input1 = inserted_op_list[item_y * parts_num_x +
                                                          item_x + eltwise_insert_pos_idx].outputs[0]
                            else:
                                input1 = graph.nodes[eltwise_input[item_y * parts_num_x + item_x]].outputs[0]
                            inputs = [inputs, input1]
                            inputs = insert_eltwise_op(inputs, each_node_work_items_out,
                                                       first_node_idx, node, inserted_op_list, item_y, item_x)
                        else:
                            inputs = insert_non_slide_window_op(inputs,
                                                                each_node_work_items_out,
                                                                all_work_items_in,
                                                                first_node_idx, node,
                                                                inserted_op_list,
                                                                item_x,
                                                                item_y)

        # ###################################################################################
        #
        # last node output.no overlap
        # current node need change name, input and output shape
        ####################################################################################

        current_node = graph.nodes[last_node_idx]

        if current_node.type == OpType.FullyConnected:
            slice_num = parts_num_y
        elif current_node.type == OpType.MatMul:
            slice_num = parts_num_y + parts_num_x
        else:
            slice_num = parts_num_y * parts_num_x
        slice_num = slice_num + eltwise_insert_pos_idx
        node_num = last_node_idx - first_node_idx + 1
        shape_dim = len(current_node.inputs[0].shape)
        axis_shift = 0
        if shape_dim == 2:
            axis_shift = 1
        if current_node.type == OpType.MatMul:
            axis_shift = 3
        elif current_node.type == OpType.Transpose:
            axis_shift = -1
        if end_eltwise_idx_list[segment] > 0 and end_eltwise_idx_list[segment] < tot_nodes_num:

            end_eltwise_idx_list[segment] = 0
            get_eltwise_input0(first_node_idx, parts_num_x, parts_num_y, node_num, eltwise_inputs)
            all_nodes_eltwise.append(np.array(eltwise_inputs))
        else:
            #########################################################################
            # there are items in only x or y direction
            if parts_num_x == 1 or parts_num_y == 1:
                # need total concat or sub-concat to reduce footprint
                if check_concat_total_or_subblock(graph, current_node, segment_items_size, segment, start_split_idx_list,
                                                  node_offset):
                    # if concat_block>1 and sram_size > 0 \
                    #     and (len(current_node.children)>=2 \
                    #     and ('kernel_x' in current_node.children[0].params or 'kernel_x' in current_node.children[1].params))\
                    #     and segment_items_size[segment+1,0]==segment_items_size[segment,0]:

                    # concat's inputs number is parts_num_y*parts_num_x
                    for item_y in range(parts_num_y - 1):
                        concat_name = current_node.name + '_item' + str(item_y) + 'to' + str(item_y + 1) + "_merge"
                        # new concat op
                        concat = PyNode(graph.get_valid_node_name(concat_name), OpType.Concat)
                        h = 0
                        for i in range(concat_block - (item_y == parts_num_y - concat_block + 1)):
                            inputs = inserted_op_list[slice_num + (item_y + i + 1) * node_num - 1].outputs[0]
                            concat.add_input(inputs)
                            h = h + inputs.ir_shape[1]
                        w = current_node.outputs[0].ir_shape[2]
                        # new concat output
                        # current node's output are as concat node's output

                        new_outp_tensor_name = graph.get_valid_tensor_name(
                            current_node.outputs[0].name + '_item_y' + str(item_y))
                        newshape = TensorShape([current_node.outputs[0].shape[0], h,
                                                w, current_node.outputs[0].shape[3]])

                        new_outp_tensor = PyTensor(new_outp_tensor_name, newshape, current_node.outputs[0].dtype)
                        new_outp_tensor.ir_shape = newshape
                        new_outp_tensor.qmin = current_node.outputs[0].qmin
                        new_outp_tensor.qmax = current_node.outputs[0].qmax
                        new_outp_tensor.similarity = current_node.outputs[0].similarity
                        concat.add_output(new_outp_tensor)
                        for i in range(len(current_node.children)):
                            current_node.children[i].add_input(new_outp_tensor)
                        concat.params['axis'] = 1
                        inserted_op_list.append(concat)
                        current_node.add_input(new_outp_tensor)
                        # sub_concat_op_list.append(concat)
                        # set  default value
                        concat.attrs.update(current_node.attrs.clone())
                else:
                    concat_name = current_node.name + '_item0' + 'to' + str(parts_num_y * parts_num_x) + "_merge"
                    # new concat op
                    concat = PyNode(graph.get_valid_node_name(concat_name), OpType.Concat)
                    # concat's inputs number is parts_num_y*parts_num_x
                    for item_y in range(parts_num_y):
                        for item_x in range(parts_num_x):
                            inputs = \
                                inserted_op_list[slice_num + (item_y * parts_num_x + item_x + 1)
                                                 * node_num - 1].outputs[0]
                            concat.add_input(inputs)

                    # new concat output
                    # current node's output are as concat node's output
                    concat.add_output(current_node.outputs[0])
                    concat.attrs.update(current_node.attrs.clone())
                    # params need is concat related
                    if parts_num_x == 1:
                        concat.params['axis'] = 1 - axis_shift
                    else:
                        concat.params['axis'] = 2 - axis_shift
                    # quantized IR, need scale and shift
                    inserted_op_list.append(concat)
            else:
                #########################################################################
                # there are items in x and y direction
                concat_name = current_node.name + '_axis_y' + "_concat"
                concat_h = PyNode(graph.get_valid_node_name(concat_name), OpType.Concat)
                for item_y in range(parts_num_y):
                    concat_name = current_node.name + '_item_x' + str(item_y) + "_concat"
                    # new concat op
                    concat = PyNode(graph.get_valid_node_name(concat_name), OpType.Concat)
                    for item_x in range(parts_num_x):
                        inputs = \
                            inserted_op_list[slice_num + (item_y * parts_num_x + item_x + 1) * node_num - 1].outputs[0]
                        concat.add_input(inputs)

                    # get w direction concat result, then take it as h direction concat input
                    if axis_shift == 1:
                        h = inputs.ir_shape[0]
                        w = current_node.outputs[0].ir_shape[1]
                        new_outp_tensor_name = graph.get_valid_tensor_name(
                            current_node.outputs[0].name + '_item_y' + str(item_y))
                        newshape = TensorShape([h, w])
                    elif axis_shift == 3:
                        h = inputs.ir_shape[-2]
                        w = current_node.outputs[0].ir_shape[-1]
                        new_outp_tensor_name = graph.get_valid_tensor_name(
                            current_node.outputs[0].name + '_item_y' + str(item_y))
                        newshape = TensorShape([current_node.outputs[0].shape[0],
                                                current_node.outputs[0].shape[1], h, w])
                    # transpose case
                    elif axis_shift == -1:
                        h = inputs.ir_shape[2]
                        w = current_node.outputs[0].ir_shape[3]
                        new_outp_tensor_name = graph.get_valid_tensor_name(
                            current_node.outputs[0].name + '_item_y' + str(item_y))
                        newshape = TensorShape([current_node.outputs[0].shape[0],
                                                current_node.outputs[0].shape[1], h, w])
                    else:
                        h = inputs.ir_shape[1]
                        w = current_node.outputs[0].ir_shape[2]
                        new_outp_tensor_name = graph.get_valid_tensor_name(
                            current_node.outputs[0].name + '_item_y' + str(item_y))
                        newshape = TensorShape([current_node.outputs[0].shape[0], h,
                                                w, current_node.outputs[0].shape[3]])
                    new_outp_tensor = PyTensor(new_outp_tensor_name, newshape, current_node.outputs[0].dtype)
                    new_outp_tensor.ir_shape = newshape
                    new_outp_tensor.qmin = current_node.outputs[0].qmin
                    new_outp_tensor.qmax = current_node.outputs[0].qmax
                    new_outp_tensor.ir_dtype = current_node.outputs[0].ir_dtype
                    new_outp_tensor.similarity = current_node.outputs[0].similarity
                    concat.add_output(new_outp_tensor)
                    concat.params['axis'] = 2 - axis_shift
                    inserted_op_list.append(concat)
                    # set  default value
                    concat.attrs.update(current_node.attrs.clone())
                    # add h dir concat input
                    concat_h.add_input(new_outp_tensor)

                #################################################################
                # set  default value
                concat_h.attrs.update(current_node.attrs.clone())
                # params need is concat related
                concat_h.params['axis'] = 1 - axis_shift

                concat_h.add_output(current_node.outputs[0])

                # quantized IR, need scale and shift
                inserted_op_list.append(concat_h)
        # graph.remove_node(graph.nodes[last_node_idx])
        tot_node = len(inserted_op_list)

        # insert new op to nodes
        for node in range(tot_node - 1, -1, -1):
            graph.nodes.insert(last_node_idx + 1, inserted_op_list[node])
            inserted_op_list[node].quantized = True
        # remove original nodes
        # to remove this layer parent's child, child is replaced by silce

        graph.nodes[first_node_idx].remove_input(graph.nodes[first_node_idx].inputs[0])
        graph.nodes[last_node_idx].remove_output(graph.nodes[last_node_idx].outputs[0])
        for node in range(first_node_idx, last_node_idx + 1, 1):
            graph.remove_node(graph.nodes[first_node_idx])

        node_offset = node_offset + tot_node - node_num

    # update each layer id
    for node in range(len(graph.nodes)):
        graph.nodes[node].attrs['layer_id'] = str(int(node))
