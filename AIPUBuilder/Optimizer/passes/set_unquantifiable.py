# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import dtype2str, str2dtype, is_float, dtype2torch_type


def set_unquantifiable(graph):
    def _check_lib_impl():
        has_float = False
        optype = node.type
        dtype_specs_from_lib = get_op_dtype_spec(optype)
        ds = set()
        for spec in dtype_specs_from_lib:
            dspec = spec.in_dtypes + spec.out_dtypes
            ds = ds | set(dspec)
        fs = set(float_list)
        candidates = list(fs.intersection(ds)) if len(ds) > 0 else float_list
        has_float = True if len(candidates) else False
        dtype = candidates[0] if len(candidates) else 'disable'
        for dt in candidates:
            idx = trigger_float_op.find('_')
            if dtype2str(dt) == trigger_float_op[:idx]:
                dtype = dt
                break
        return has_float, dtype

    float_list = [Dtype.FP16, Dtype.BFP16, Dtype.FP32]
    for node in graph.nodes:
        trigger_float_op = str(node.attrs['trigger_float_op']).lower().strip()
        is_lib_float = False
        if 'disable' != trigger_float_op:
            is_lib_float, dt = _check_lib_impl()
        node.params['unquantifiable'] = is_lib_float
        node.attrs['trigger_float_op'] = dt if is_lib_float else 'disable'
