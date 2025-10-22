# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
from .set_unquantifiable import set_unquantifiable


def control_op_preprocess(graph: PyGraph, config):
    inserted_ops = graph.insert_dummy_node_ahead(
        OpType.Dummy, condition_func=lambda node, parent_node, edge_tensor: (parent_node.type in [OpType.If, OpType.Loop]))

    trigger_float_op_cfg = 'float16_preferred!' if 'disable' == config.trigger_float_op.get() else config.trigger_float_op.get()
    if '!' not in trigger_float_op_cfg:
        trigger_float_op_cfg = trigger_float_op_cfg + '!'
    for dn in inserted_ops:
        pn = dn.parents[0]
        # control operator has subgraph, which is not suitable for quantization
        if 'disable' == pn.attrs['trigger_float_op']:
            pn.attrs['trigger_float_op'] = trigger_float_op_cfg
            dn.attrs['trigger_float_op'] = trigger_float_op_cfg
    for n in graph.nodes:
        if len(n.children) < 1 and n.type in [OpType.If, OpType.Loop]:
            if 'disable' == n.attrs['trigger_float_op']:
                n.attrs['trigger_float_op'] = trigger_float_op_cfg
    for _, sg in graph.subgraph_map.items():
        for sn in sg.nodes:
            tkey = sn.type.name.lower()
            sn.attrs['trigger_float_op'] = trigger_float_op_cfg if tkey not in config.trigger_float_op.tdict.keys(
            ) else config.trigger_float_op.tdict[tkey]
            if sn.type in [OpType.If, OpType.Loop]:
                sn.attrs['trigger_float_op'] = trigger_float_op_cfg
            sn.attrs['q_mode_weight'] = config.quantize_method_for_weight.get(
            ) if tkey not in config.quantize_method_for_weight.tdict.keys() else config.quantize_method_for_weight.tdict[tkey]
            sn.attrs['q_bits_weight'] = config.weight_bits.get(
            ) if tkey not in config.weight_bits.tdict.keys() else config.weight_bits.tdict[tkey]
            sn.attrs['weight_block_size'] = config.weight_block_size.get(
            ) if tkey not in config.weight_block_size.tdict.keys() else config.weight_block_size.tdict[tkey]
            sn.attrs["min_compatible_zhouyi_target"] = config.min_compatible_zhouyi_target

        set_unquantifiable(sg, config)
        sg.set_fp_tensor_after_update_quantization_attr()
        cast_dtypes_for_lib = config.cast_dtypes_for_lib if config is not None and hasattr(
            config, 'cast_dtypes_for_lib') else False

        op_need_cast_dtypes_for_lib = set()
        if isinstance(cast_dtypes_for_lib, bool):
            op_need_cast_dtypes_for_lib = set([sn for sn in sg.nodes]
                                              ) if cast_dtypes_for_lib else set()
        else:
            for sn in sg.nodes:
                tkey = sn.type.name.lower()
                mkey = sn.params['method'].lower() if (sn.type == OpType.Activation) else ''
                tdict = cast_dtypes_for_lib.tdict
                if (tkey in tdict.keys() and tdict[tkey]) or (mkey in tdict.keys() and tdict[mkey]) or cast_dtypes_for_lib.global_value:
                    op_need_cast_dtypes_for_lib.add(sn)

        if len(op_need_cast_dtypes_for_lib) > 0:
            OPT_INFO(f"These OPs will automatically cast dtypes to adapt to lib's dtypes' spec "
                     f"(may cause model accuracy loss due to corresponding spec's restriction): "
                     f"{str(op_need_cast_dtypes_for_lib)}")
            for sn in sg.nodes:
                if sn.type == OpType.Cast:
                    op_need_cast_dtypes_for_lib.discard(sn)
        sg.op_need_cast_dtypes_for_lib = op_need_cast_dtypes_for_lib
