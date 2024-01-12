# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import dtype2str, str2dtype, is_float, dtype2torch_type


def set_unquantifiable(graph):
    def _check_lib_impl():
        has_float = False
        for inp in node.inputs:
            if is_float(inp.ir_dtype):
                has_float = True
                break
        for out in node.outputs:
            if is_float(out.ir_dtype):
                has_float = True
                break
        for name, const in node.constants.items():
            if 'weight' in name or 'bias' in name:
                if is_float(const.ir_dtype):
                    has_float = True
                    break
        if not has_float:
            return False, None
        dtype_specs_from_lib = node.get_lib_dtype_spec()
        ds = set()
        for spec in dtype_specs_from_lib:
            dspec = spec.in_dtypes + spec.out_dtypes
            ds = ds | set(dspec)
        fs = set(float_list)
        candidates = list(fs.intersection(ds)) if len(ds) > 0 else float_list
        has_float = True if len(candidates) else False
        dtype = None
        if len(candidates) > 0:
            dtype = candidates[0] if Dtype.FP16 not in candidates else Dtype.FP16
        for dt in candidates:
            idx = trigger_float_op.find('_')
            if dtype2str(dt) == trigger_float_op[:idx]:
                dtype = dt
                break
        return has_float, dtype

    float_list = [Dtype.FP16, Dtype.BFP16, Dtype.FP32]
    for node in graph.nodes:
        trigger_float_op = str(node.attrs['trigger_float_op']).lower().strip()
        node.attrs['ori_trigger_float_op'] = trigger_float_op
        is_lib_float = False
        if 'disable' != trigger_float_op:
            if '!' in trigger_float_op:
                idx = trigger_float_op.find('_')
                is_lib_float, dt = True, str2dtype(trigger_float_op[:idx])
            else:
                is_lib_float, dt = _check_lib_impl()
        node.params['unquantifiable'] = is_lib_float
        if is_lib_float:
            if 'int' in trigger_float_op:
                node.attrs['weight_only_quantization'] = True
            node.attrs['trigger_float_op'] = dt
        else:
            node.attrs['trigger_float_op'] = 'disable'

    """
    if constant op is not unquantifiable and its layer_top_type_original is float type, but 
    its children nodes have unquantifiable==true, constant op would be setted params['unquantifiable']=true.
    """
    for n in graph.nodes:
        if n.type in [OpType.Constant] and not n.params['unquantifiable']:
            top_type_original = n.attrs['layer_top_type_original'][0]
            original_top_dtype = str2dtype(top_type_original)
            is_original_top_float = is_float(original_top_dtype)
            if not is_original_top_float:
                continue
            for cn in n.children:
                if cn.params['unquantifiable']:
                    n.params['unquantifiable'] = True
                    if 'int' in n.attrs['ori_trigger_float_op']:
                        node.attrs['weight_only_quantization'] = True
                    n.attrs['trigger_float_op'] = cn.attrs['trigger_float_op']
                    break
