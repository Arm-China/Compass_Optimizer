# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
from . shrink_pow_exponent_s1 import shrink_pow_exponent_s1
from . set_unquantifiable import set_unquantifiable
from . unify_scales_for_multi_inputs_operator import opt_unify_scales_for_multi_inputs_operators
from . insert_op import (InsertCastOp,
                         InsertQuantizeOp,
                         InsertDeQuantizeOp,
                         InsertPadOp)
from .convert_resize_to_convolution import convert_resize_to_convolution
# =========================optimization stage 1 passes============================================


def optimization_stage1(graph, config):
    shrink_pow_exponent_s1(graph, config)
    # update_unquantifiable(graph)
# =========================optimization stage 1 end   ============================================


# =========================optimization stage 2 passes============================================
def optimization_stage2(graph, config):
    pass
# =========================optimization stage 2 end   ============================================

# =========================optimization stage 3 passes============================================


def optimization_stage3(graph, config):
    #  insert pad op for avgpool when count_include_pad=ceil_mode=True zp!=0
    _insert_obj = [InsertPadOp]
    for _obj in _insert_obj:
        handler = _obj(graph, config)
        handler.run()
    pass
# =========================optimization stage 3 end   ============================================


# =========================before graph quantize passes===========================================
def insert_op_pass(graph, config, insert_obj):
    graph.set_tensor_quantization_attrs()

    set_unquantifiable(graph)
    for _obj in insert_obj:
        handler = _obj(graph, config)
        handler.run()
    graph.clear_tensor_quantization_attrs()


def adapt_float_subgraph_pass(graph, config):
    from AIPUBuilder.Optimizer.ops.conv import linear_op_quantize
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
        if qn.type == OpType.Quantize and hasattr(qn, 'additional') and qn.additional:
            ot = qn.outputs[0]
            qinfo = qn.attrs['qinfo']
            for qk, qv in qinfo.items():
                ot.__setattr__(qk, qv)

            if 'quantize_scale' in qn.params:
                ot.scale = qn.params['quantize_scale']
                ot.zerop = qn.params['quantize_zp']
            else:
                ot.scale = qn.constants['quantize_scale'].betensor
                ot.zerop = qn.constants['quantize_zp'].betensor
            if len(qn.children) == 1 and qn.children[0].type == OpType.Cast:
                cn = qn.children[0].clone()
                cn.quantize()
                ct = cn.outputs[0]
                ot.dtype = ct.dtype
                ot.scale = ct.scale
                ot.zerop = ct.zerop
                ot.qinvariant = ct.qinvariant
                ot.qmin = ct.qmin
                ot.qmax = ct.qmax
                ot.qbits = ct.qbits
                if 'quantize_scale' in qn.params:
                    qn.params['quantize_scale'] = ct.scale
                    qn.params['quantize_zp'] = ct.zerop
                else:
                    qn.constants['quantize_scale'].betensor = ct.scale
                    qn.constants['quantize_zp'].betensor = ct.zerop

        fd = qn.attrs['trigger_float_op'].name if Dtype == type(
            qn.attrs['trigger_float_op']) else str(qn.attrs['trigger_float_op']).lower().strip()
        if fd != 'disable' and qn.params['unquantifiable']:
            # because graph.clear_tensor_quantization_attrs will change the dtype to tensor.betensor.dtype, so we will
            # explicit to change the output tensor to trigger_float_op (has changed to one of [float16, bfloat16, float32])
            o_dtype = str2dtype(fd)
            for ot in qn.outputs:
                if is_float(ot.dtype):
                    ot.dtype = o_dtype
            for key, ct in qn.constants.items():
                if is_float(ct.dtype):
                    ct.dtype = Dtype.FP32 if 'biases' == key else o_dtype

            if "weights" in qn.constants.keys() and qn.get_attrs("weight_only_quantization", optional=True, default_value=False):
                # use linear_op_quantize for dealing unify_shifts_for_aiff with per-n-channel quantization
                wn = qn.clone(qn.name+"_clone_")
                wn.graph = None
                wn.params['unquantifiable'] = False
                if len(wn.inputs) > 0:
                    wn.inputs[0].scale = 1.0
                    wn.inputs[0].zerop = 0
                else:
                    # constant node
                    it = PyTensor('tmp_input')
                    it.scale = 1.0
                    it.zerop = 0
                    wn.add_input(it)
                # make sure out.scale == 1.0, so out.scale/(inp.scale * wht.scale) == 1.0/wht.scale
                wn.attrs['q_bits_activation'] = max(qn.attrs['q_bits_activation'], 13)
                wn.attrs["q_mode_activation"] = QuantMode.to_per_tensor(
                    QuantMode.to_asymmetric(wn.attrs["q_mode_activation"]))
                wn.params['with_activation'] = 'none'
                wn.outputs[0].min, wn.outputs[0].max = bits2range(wn.attrs['q_bits_activation'], True)
                linear_op_quantize(wn)
                qn.constants["weights"] = wn.constants["weights"]
                if "scale" in wn.constants.keys():
                    qn.constants["scale"] = wn.constants["scale"]
                else:
                    qn.params['scale_value'] = wn.params['scale_value']
                    qn.params['scale_type'] = wn.params['scale_type']
                if "shift" in wn.constants.keys():
                    qn.constants["shift"] = wn.constants["shift"]
                else:
                    qn.params['shift_value'] = wn.params['shift_value']
                    qn.params['shift_type'] = wn.params['shift_type']
                # mark on IR, weights compressed through weight only quantization (may support other compress methods in the future)
                qn.params['weight_compressed'] = True


def unify_scales_for_multi_inputs_op_pass(graph, config):
    if config.unify_scales_for_concat:
        # historical compatibility
        optype_cfg_dt = {}
        optype_cfg_dt[OpType.Concat] = (config.unify_scales_for_concat_max_depth,
                                        config.unify_scales_for_concat_threshold, 'out')
        OPT_INFO(
            f"trying to unify scales for concat's branches based on statistic info, now the config is: {optype_cfg_dt}")
        opt_unify_scales_for_multi_inputs_operators(graph, optype_cfg_dt)
    if config.unify_scales_for_multi_inputs_operators:
        from AIPUBuilder.Optimizer.config.cfg_fields import UnifyScales4MultiInputsOP
        optype_cfg_dt = UnifyScales4MultiInputsOP.parse(config.unify_scales_for_multi_inputs_operators)
        OPT_INFO(
            f"trying to unify input branches' scales for assigned operators based on statistic info, now the config is: {optype_cfg_dt}")
        opt_unify_scales_for_multi_inputs_operators(graph, optype_cfg_dt)

# =========================before graph quantize end  ============================================
