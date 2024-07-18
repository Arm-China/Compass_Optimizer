# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
from . shrink_pow_exponent_s1 import shrink_pow_exponent
from . merge_matmul_mul_s1 import merge_matmul_mul
from . convert_resize_to_convolution import convert_resize_to_convolution
from . decompose_nonmonotonic_activations_s1 import decompose_nonmonotonic_activations
from . tune_op_extra_params_s1 import *
from . check_quantization_info_s1 import check_quantization_info
from . split_qkv_fc_s1 import split_qkv_fc
from . detect_inf_mask_nodes import detect_inf_mask_nodes
from . set_unquantifiable import set_unquantifiable
from . unify_scales_for_multi_inputs_operator import opt_unify_scales_for_multi_inputs_operators
from . optimize_x2_wdc import optimize_x2_wdc
from . global_calibration_prepare import global_calibration_prepare
from . insert_op import (InsertCastOp,
                         InsertQuantizeOp,
                         InsertDeQuantizeOp,
                         InsertPadOp)
from . absorb_cast_to_clip import absorb_cast_to_clip

# =========================optimization stage 1 passes============================================


def optimization_stage1(graph, config):
    shrink_pow_exponent(graph, config)
    if config.enable_pass_decompose_nonmonotonic_activations:
        decompose_nonmonotonic_activations(graph, config)
    if config.enable_pass_tune_op_complicated_activations:
        tune_op_complicated_activations(graph, config)
        tune_op_trigonometric_activations(graph, config)
    if config.enable_pass_tune_op_softmax:
        tune_op_softmax(graph, config)
    if config.enable_pass_merge_matmul_mul:
        merge_matmul_mul(graph, config)

    check_quantization_info(graph, config)
    for node in graph.nodes:
        for ot in node.outputs:
            ot.key_axis = None
    global_calibration_prepare(graph, config)
    detect_inf_mask_nodes(graph, config)
    split_qkv_fc(graph, config)

# =========================optimization stage 1 end   ============================================


# =========================optimization stage 2 passes============================================
def optimization_stage2(graph, config):
    optimize_x2_wdc(graph, config)
# =========================optimization stage 2 end   ============================================

# =========================optimization stage 3 passes============================================


def optimization_stage3(graph, config):
    # revert Matmul (with additional param: fused_multiplier) into Matmul+Mul if Matmul is unquantifiable
    if config.enable_pass_merge_matmul_mul:
        def condition_func(node, parent_node, edge_tensor): return (
            (OpType.MatMul == parent_node.type) and parent_node.params['unquantifiable'] and (
                'fused_multiplier' in parent_node.attrs)
        )
        inserted_nodes = graph.insert_dummy_node_ahead(OpType.Mul, condition_func)
        for n in inserted_nodes:
            n.params['unquantifiable'] = True
            parent_node = n.parents[0]
            n.attrs.update(parent_node.attrs.clone())
            fused_multiplier = parent_node.attrs['fused_multiplier']
            if 'fused_multiplier' in parent_node.params:
                parent_node.params.pop('fused_multiplier')
            ft = PyTensor(graph.get_valid_tensor_name(n.inputs[0].name + '_scale'), fused_multiplier)
            ft.dtype = n.inputs[0].dtype
            fn = PyNode(graph.get_valid_node_name(parent_node.name + '_scale'), OpType.Constant)
            fn.attrs.update(n.attrs.clone())
            fn.constants['weights'] = ft.clone(graph.get_valid_tensor_name(ft.name + '_w'))
            fn.params['unquantifiable'] = True
            fn.additional = True
            fn.add_output(ft)
            n.add_input(ft)
            graph.add_node(fn)

    absorb_cast_to_clip(graph, config)
# =========================optimization stage 3 end   ============================================


# =========================before graph quantize passes===========================================
def insert_op_pass(graph, config, insert_obj):
    graph.set_tensor_quantization_attrs()
    set_unquantifiable(graph, config)
    for _obj in insert_obj:
        _obj(graph, config)()
    graph.clear_tensor_quantization_attrs()


def adapt_float_subgraph_pass(graph, config):
    '''
    now this pass mainly:
    - update QuantizeOp quantization info from node.params to node.outputs[0] tensor and inputs tensor dtype
    - update DeQuantizeOp outputs tensor dtype
    - update op outputs tensor dtype in unquantifiable == True op

    and the above tensor dtype update will delete when forward supports fp16/bf16 forward.
    '''
    # graph is a quantized graph
    # if config.trigger_float_op.lower() != 'disable':
    for qn in graph.nodes:
        if qn.type in [OpType.Quantize, OpType.DeQuantize] and hasattr(qn, 'additional') and qn.additional:
            ot = qn.outputs[0]
            if 'quantize_scale' in qn.attrs and 'quantize_zp' in qn.attrs:
                # dequantize op ot.scale/ot.zerop is 1.0 and 0
                if qn.type == OpType.Quantize:
                    ot.scale = qn.attrs['quantize_scale']
                    ot.zerop = qn.attrs['quantize_zp']
                qn.set_ir_field('quantize_scale', qn.attrs['quantize_scale'], Dtype.FP32)
                qn.set_ir_field('quantize_zp', qn.attrs['quantize_zp'], Dtype.INT32)
                if 'quantize_scale' in qn.constants:
                    qn.constants['quantize_scale'].attrs['unchanged_data'] = True
                    qn.constants['quantize_zp'].attrs['unchanged_data'] = True
            else:
                OPT_WARN(f"{qn} needs quantize_scale and quantize_zp in node attrs, please check it.")

        if qn.type == OpType.Quantize and hasattr(qn, 'additional') and qn.additional:
            ot = qn.outputs[0]
            qinfo = qn.attrs['qinfo']
            for qk, qv in qinfo.items():
                ot.__setattr__(qk, qv)

            if len(qn.children) == 1 and qn.children[0].type == OpType.Cast:
                cn = qn.children[0].clone()
                cn.quantize()
                ct = cn.outputs[0]
                ot.clone_qinfo(ct)
                if 'quantize_scale' in qn.params:
                    qn.params['quantize_scale'] = ct.scale[0]
                    qn.params['quantize_zp'] = ct.zerop[0]
                else:
                    qn.constants['quantize_scale'].betensor = ct.scale
                    qn.constants['quantize_zp'].betensor = ct.zerop

        fd = qn.attrs['trigger_float_op'].name if isinstance(qn.attrs['trigger_float_op'], Dtype) \
            else str(qn.attrs['trigger_float_op']).lower().strip()
        if fd != 'disable' and qn.params['unquantifiable']:
            # because graph.clear_tensor_quantization_attrs will change the dtype to tensor.betensor.dtype, so we will
            # explicit to change the output tensor to trigger_float_op (has changed to one of [float16, bfloat16, float32])
            o_dtype = str2dtype(fd)
            for ot in qn.outputs:
                if is_float(ot.dtype):
                    ot.dtype = o_dtype
            if qn.type == OpType.Cast:
                qn.params['to_dtype'] = qn.outputs[0].dtype
            for key, ct in qn.constants.items():
                if is_float(ct.dtype) and ('unchanged_data' not in ct.attrs or not ct.attrs['unchanged_data']):
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


def unify_scales_for_multi_inputs_op_pass(graph, config):
    if config.unify_scales_for_concat:
        # historical compatibility
        optype_cfg_dt = {}
        optype_cfg_dt[OpType.Concat] = (config.unify_scales_for_concat_max_depth,
                                        config.unify_scales_for_concat_threshold, 'out')
        OPT_INFO(
            f"trying to unify scales for concat's branches based on statistic info, now the config is: {optype_cfg_dt}")
        opt_unify_scales_for_multi_inputs_operators(graph, optype_cfg_dt)
    if config.__getattr__('unify_scales_for_multi_inputs_operators'):
        OPT_INFO(
            f"trying to unify input branches' scales for assigned operators based on statistic info")
        opt_unify_scales_for_multi_inputs_operators(graph, None)

# =========================before graph quantize end  ============================================
