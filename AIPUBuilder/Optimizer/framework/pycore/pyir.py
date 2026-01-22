# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3


def cast_to_NodeParamValue(v):
    from AIPUBuilder.Optimizer.logger import OPT_WARN
    try:
        from AIPUBuilder._C._core import NodeParamValue
        from AIPUBuilder._C._core import Dtype as _cDtype
    except Exception as e:
        OPT_WARN(
            f"when calling cast_to_NodeParamValue(c data struct), please install AIPUBuilder package. now error message: {e}", log_once=True)
    from AIPUBuilder.Optimizer.framework.pycore.pytype import Dtype as _pyDtype
    import torch
    import numpy as np

    dt_dict = {str(t[0].name): t[0] for t in _cDtype.__entries.values()}

    if isinstance(v, _pyDtype):
        v = dt_dict[v.name]
    elif isinstance(v, bool):
        v = v
    elif isinstance(v, (int, float, str, list)):
        v = NodeParamValue(v)
    elif isinstance(v, torch.Tensor) and (v.dim() < 1 or v.numel() == 1):
        v = NodeParamValue(v.reshape(1).item())
    elif isinstance(v, np.ndarray) and v.ndim < 1:
        v = NodeParamValue(v.item())
    else:
        try:
            v = NodeParamValue(v)
        except Exception as e:
            v = str(v)
    return v


def cast_from_NodeParamValue_string(v):
    from AIPUBuilder.Optimizer.utils.dtype_utils import str2dtype, dtype2str
    from AIPUBuilder.Optimizer.framework.pycore.pytype import Dtype
    import re

    def is_valid_list_string(sv):
        if (len(sv) > 1) and (sv[0] == '[') and (sv[-1] == ']'):
            stk = []
            for i in range(len(sv)):
                c = sv[i]
                if '[' == c:
                    stk.append(c)
                elif ']' == c:
                    if len(stk) < 1:
                        return False
                    else:
                        stk.pop()
            if len(stk) < 1:
                return True
            else:
                return False
        else:
            return False

    dt_s = ''
    for dt in Dtype:
        str_dt = dtype2str(dt)
        dt_s += '|'.join([str_dt, f"dtype.{str_dt}|"])
    sv = v.strip()
    if is_valid_list_string(sv):
        # list
        lt = []
        ts = sv[1:-1] + ','
        cntl = 0
        cntr = 0
        pos = 0
        for i in range(len(ts)):
            c = ts[i]
            if '[' == c:
                cntl += 1
            elif ']' == c:
                cntr += 1
            elif ',' == c:
                if cntl == cntr:
                    sub_str = ts[pos:i].strip(' \'')
                    if len(sub_str) > 0:
                        lt.append(cast_from_NodeParamValue_string(sub_str))
                    pos = i+1
        return lt
    elif re.match('^('+dt_s[:-1]+')$', sv.lower()):
        return str2dtype(sv)
    elif re.match(r'^(\-|\+)?\d+$', sv.lower()):
        return int(sv)
    elif re.match(r'^(((\-|\+)?\d+((\.\d+)|\.)?(e(\-|\+)?\d+)?)|((\-|\+)?inf))$', sv.lower()):
        return float(sv)
    elif re.match(r'^true$', sv.lower()):
        return True
    elif re.match(r'^false$', sv.lower()):
        return False
    else:
        # str
        return str(sv)


def cast_to_NodeParamValue_string(v):
    from AIPUBuilder.Optimizer.framework.pycore.pygraph import PyGraph
    from AIPUBuilder.Optimizer.framework.pycore.pytype import Dtype
    from AIPUBuilder.Optimizer.utils.dtype_utils import dtype2str
    import torch
    import numpy as np
    if isinstance(v, Dtype):
        return dtype2str(v)
    elif isinstance(v, bool):
        return str(v).lower()
    elif isinstance(v, int):
        return str(v)
    elif isinstance(v, float):
        return str(v)
    elif isinstance(v, str):
        return v
    elif isinstance(v, torch.Tensor) and (v.dim() < 1 or v.numel() == 1):
        return str(v.reshape(1).item())
    elif isinstance(v, np.ndarray) and v.ndim < 1:
        return str(v.item())
    elif isinstance(v, PyGraph):
        return str(v.name)
    elif v is None:
        return str(v)
    else:
        # list
        sv = '['
        cnt = 0
        for e in v:
            sv += cast_to_NodeParamValue_string(e) + ','
            cnt += 1
        if cnt > 0:
            return sv[:-1] + ']'
        else:
            return sv + ']'


def convert_aipu_node_to_opt_node(cn):
    from AIPUBuilder.Optimizer.framework.qgraph import QuantizeGraph
    from AIPUBuilder.Optimizer.framework.pycore.pynode import PyNode
    from AIPUBuilder.Optimizer.framework.pycore.pytensor import PyTensor, TensorShape
    from AIPUBuilder.Optimizer.framework.pycore.pytype import Dtype, OpTypeValue
    from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
    from AIPUBuilder.Optimizer.utils import dtype2range, is_torch_tensor_with_multi_data

    dt_dict = {dt.name: dt for dt in Dtype}
    pn = PyNode(cn.name, OpTypeValue(str(cn.type)))
    pn.attrs['layer_id'] = str(cn.attrs.get("layer_id", 0))
    for k, v in cn.params.items():
        # when scale_type is a list,like scale_type=[uint16, uint16 uint16] in eltwise, and actuallly
        # scale_type[0] is a _C.Dtype, when str(scale_type), it is
        # '[<Dtype.UINT16: 4>, <Dtype.UINT16: 4>, <Dtype.UINT16: 4>]'
        # and cast_from_NodeParamValue_string() cannot parse these.
        vv = str(v) if not isinstance(v, (list, tuple)) else str([str(ve) for ve in v])
        pn.params[k] = cast_from_NodeParamValue_string(str(vv))
    for k, v in cn.constants.items():
        pv = PyTensor(v.name, v.data())
        pv.dtype = dt_dict[v._dtype().__to_str__().upper()]
        pv.ir_dtype = pv.dtype
        pv.ir_shape = TensorShape(tuple(v.shape))
        pv.scale = v.quantization.scales if len(v.quantization.scales) > 0 else 1.0
        pv.zerop = v.quantization.offsets if len(v.quantization.offsets) > 0 else 0
        pn.constants[k] = pv
        if v.quantization.quantized:
            pv.qmin, pv.qmax = dtype2range(pv.ir_dtype)
            pv.qbits = v.quantization.bits
            pv.qinvariant = v.quantization.qinvariant
        if 'range' in v._attrs:
            pv.ir_range = v._attrs['range']
    for v in cn.inputs:
        pv = PyTensor(v.name, v.data())
        pv.is_act = True
        pv.dtype = dt_dict[v._dtype().__to_str__().upper()]
        pv.ir_dtype = pv.dtype
        pv.ir_shape = tuple(v.shape)
        pv.scale = v.quantization.scales if len(v.quantization.scales) > 0 else 1.0
        pv.zerop = v.quantization.offsets if len(v.quantization.offsets) > 0 else 0
        pv.key_axis = None if not is_torch_tensor_with_multi_data(pv.scale) else -1  # qtlib not key_axis, so tmp -1
        if v.quantization.quantized:
            pv.qmin, pv.qmax = dtype2range(pv.ir_dtype)
            pv.qbits = v.quantization.bits
            pv.qinvariant = v.quantization.qinvariant
            pv.min = v.quantization.mins
            pv.max = v.quantization.maxs
        if 'range' in v._attrs:
            pv.ir_range = v._attrs['range']
        pn.add_input(pv)
    for v in cn.outputs:
        pv = PyTensor(v.name, v.data())
        pv.is_act = True
        pv.dtype = dt_dict[v._dtype().__to_str__().upper()]
        pv.ir_dtype = pv.dtype
        pv.ir_shape = tuple(v.shape)
        pv.scale = v.quantization.scales if len(v.quantization.scales) > 0 else 1.0
        pv.zerop = v.quantization.offsets if len(v.quantization.offsets) > 0 else 0
        pv.key_axis = None if not is_torch_tensor_with_multi_data(pv.scale) else -1  # qtlib not key_axis, so tmp -1
        if v.quantization.quantized:
            pv.qmin, pv.qmax = dtype2range(pv.ir_dtype)
            pv.qbits = v.quantization.bits
            pv.qinvariant = v.quantization.qinvariant
            pv.min = v.quantization.mins
            pv.max = v.quantization.maxs
        if 'range' in v._attrs:
            pv.ir_range = v._attrs['range']
        pn.add_output(pv)
    PyNode.deduce_quantization_infos(pn)
    return pn


def ds_expr_to_str(exprs):
    from AIPUBuilder._C._tongue import Symbol, Expr
    if isinstance(exprs, str):
        return exprs

    if isinstance(exprs, (Expr, Symbol)):
        str_expr = str(exprs)
        if str_expr.isdigit():
            str_expr = int(str_expr)
        return str_expr

    if isinstance(exprs, (list, tuple)):
        expr_str = []
        for expr in exprs[:]:
            expr_str.append(ds_expr_to_str(expr))
        return expr_str


def str_to_ds_expr(exprs):
    from AIPUBuilder._C._tongue import Symbol, Expr
    if isinstance(exprs, (int, str)):
        return Expr(exprs)
    # if isinstance(exprs, str):
    #     if '*' in exprs:
    #         exprs = '(' + exprs + ')'
    #     return Expr(exprs)

    if isinstance(exprs, (Expr, Symbol)):
        return exprs

    if isinstance(exprs, (list, tuple)):
        expr_str = []
        for expr in exprs[:]:
            expr_str.append(str_to_ds_expr(expr))
        return expr_str


def convert_aipu_graph_to_opt_graph(cg):
    from AIPUBuilder.Optimizer.framework.qgraph import QuantizeGraph
    from AIPUBuilder.Optimizer.framework.pycore.pynode import PyNode
    from AIPUBuilder.Optimizer.framework.pycore.pytensor import PyTensor, TensorShape
    from AIPUBuilder.Optimizer.framework.pycore.pytype import Dtype, OpTypeValue
    from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
    from AIPUBuilder.Optimizer.utils import dtype2range, is_torch_tensor_with_multi_data, dtype2bits, is_float

    g = QuantizeGraph()
    g.name = cg.name
    dt_dict = {dt.name: dt for dt in Dtype}
    nmap = {}
    emap = {}
    for n in cg.nodes:
        pn = PyNode(n.name, OpTypeValue(str(n.type)))
        pn.attrs['layer_id'] = str(n.attrs.get("layer_id", cg.nodes.index(n)))
        for k, v in n.params.items():
            # when scale_type is a list,like scale_type=[uint16, uint16 uint16] in eltwise, and actuallly
            # scale_type[0] is a _C.Dtype, when str(scale_type), it is
            # '[<Dtype.UINT16: 4>, <Dtype.UINT16: 4>, <Dtype.UINT16: 4>]'
            # and cast_from_NodeParamValue_string() cannot parse these.
            if k.startswith('ds_'):
                pn.params[k] = ds_expr_to_str(v)
                continue
            vv = str(v) if not isinstance(v, (list, tuple)) else str([str(ve) for ve in v])
            pn.params[k] = cast_from_NodeParamValue_string(str(vv))

        for k, v in n.constants.items():
            dtype = dt_dict[v._dtype().__to_str__().upper()]
            pv = PyTensor(v.name, v.numpy(), dtype)
            pv.ir_dtype = pv.dtype
            pv.ir_shape = TensorShape(tuple(v.shape))
            pv.scale = v.quantization.scales if len(v.quantization.scales) > 0 else 1.0
            pv.zerop = v.quantization.offsets if len(v.quantization.offsets) > 0 else 0
            pn.constants[k] = pv
            if v.quantization.quantized:
                pv.qmin, pv.qmax = dtype2range(pv.ir_dtype)
                pv.qbits = v.quantization.bits
                pv.qinvariant = v.quantization.qinvariant
            if 'range' in v._attrs:
                pv.ir_range = v._attrs['range']
        g.nodes.append(pn)
        pn.graph = g
        nmap[pn.name] = pn
        # store edges
        for v in n.outputs:
            dtype = dt_dict[v._dtype().__to_str__().upper()]
            if dtype2bits(dtype) <= 8 and is_float(dtype):
                # torch.float8 only valid on cpu or specific gpu, and only valid for few operators
                pv = PyTensor(v.name, v.betensor.bfloat16(), dtype)
            else:
                pv = PyTensor(v.name, v.betensor, dtype)
            # pv = PyTensor(v.name, v.betensor)
            pv.is_act = True
            # pv.dtype = dt_dict[v._dtype().__to_str__().upper()]
            pv.ir_dtype = pv.dtype
            pv.ir_shape = TensorShape(tuple(v.shape))
            pv.scale = v.quantization.scales if len(v.quantization.scales) > 0 else 1.0
            pv.zerop = v.quantization.offsets if len(v.quantization.offsets) > 0 else 0
            pv.key_axis = None if not is_torch_tensor_with_multi_data(pv.scale) else -1  # qtlib not key_axis, so tmp -1
            if v.quantization.quantized:
                pv.qmin, pv.qmax = dtype2range(pv.ir_dtype)
                pv.qbits = v.quantization.bits
                pv.qinvariant = v.quantization.qinvariant
                pv.min = v.quantization.mins
                pv.max = v.quantization.maxs
            if 'range' in v._attrs:
                pv.ir_range = v._attrs['range']
            emap[pv.name] = pv

    # connect edges
    for i, n in enumerate(cg.nodes):
        pn = g.nodes[i]
        tlist = []
        for v in n.inputs:
            tlist.append(emap[v.name])
        pn.inputs = tuple(tlist)
        nlist = []
        for x in n.parents:
            nlist.append(nmap[x.name])
        pn.parents = tuple(nlist)
        tlist = []
        for v in n.outputs:
            tlist.append(emap[v.name])
        pn.outputs = tuple(tlist)
        nlist = []
        for x in n.children:
            nlist.append(nmap[x.name])
        pn.children = tuple(nlist)

        for ak, av in n.attrs.items():
            pn.attrs[ak] = cast_from_NodeParamValue_string(str(av))
    ilist = []
    for t in cg.input_tensors:
        ilist.append(emap[t.name])
    g.input_tensors = tuple(ilist)
    olist = []
    for t in cg.output_tensors:
        olist.append(emap[t.name])
    g.output_tensors = tuple(olist)
    g.init_networkx()
    return g


def set_default_layout(tensor_shape):
    from AIPUBuilder._C._core import DataLayout
    if tensor_shape.layout == DataLayout.Flat:
        if len(tensor_shape) == 4:
            tensor_shape.layout = DataLayout.NHWC


def convert_opt_graph_to_aipu_graph(g):
    from AIPUBuilder.core import Tensor as _Tensor
    from AIPUBuilder._C._core import Graph as _Graph
    from AIPUBuilder._C._core import TensorList as _TensorList
    from AIPUBuilder._C._core import TensorShape as _TensorShape
    from AIPUBuilder._C._core import Node as _Node
    from AIPUBuilder._C._core import OpType as _OpType
    from AIPUBuilder._C._core import Dtype as _Dtype
    from AIPUBuilder._C._core import _py_register_optype
    from AIPUBuilder.Optimizer.utils.dtype_utils import dtype2torch_type
    from AIPUBuilder.Optimizer.framework import OpType

    def _register_optype(t: str):
        if t not in _OpType.__entries:
            v = _OpType(_py_register_optype(t))
            setattr(_OpType, t, _OpType(v))
            _OpType.__entries[t] = (v, None)  # (value, doc string)
        else:
            return _OpType.__entries[t][0]
        return v
    ot_dict = {str(t[0]): t[0] for t in _OpType.__entries.values()}
    dt_dict = {t[0].__to_str__().upper(): t[0] for t in _Dtype.__entries.values()}
    cg = _Graph(g.name)
    # cg.name = g.name
    # store edges
    emap = {}
    for n in g.nodes:
        for i, v in enumerate(n.outputs):
            ct = _Tensor(v.name, _TensorShape(list(v.ir_shape)), dt_dict[v.dtype.name])
            # ct._set_dtype(dt_dict[v.dtype.name])
            ct.quantization.scales = v.scale.flatten().cpu()
            ct.quantization.offsets = v.zerop.flatten().cpu()
            ct.quantization.bits = v.qbits if v.qbits else n.attrs['q_bits_activation'] if 'q_bits_activation' in n.attrs else -1
            ct.quantization.qinvariant = v.qinvariant
            if 'layer_top_range' in n.params:
                tmins = n.params['layer_top_range'][i][0]
                tmaxs = n.params['layer_top_range'][i][1]
                ct.quantization.mins = tmins if isinstance(tmins, list) else [tmins]
                ct.quantization.maxs = tmaxs if isinstance(tmaxs, list) else [tmaxs]
            # elif v.ir_range is not None:
            #     ct.quantization.mins = [v.ir_range[0]]
            #     ct.quantization.maxs = [v.ir_range[1]]
            emap[ct.name] = ct

        if 'layer_top_range' in n.params:
            n.params.pop('layer_top_range')
    has_ds = False
    for n in g.nodes:
        if n.type == OpType.Input and 'ds_output_shape' in n.params:
            has_ds = True
            from AIPUBuilder import ops
            with cg:
                c_out = ops.input(_TensorShape(list(n.outputs[0].ir_shape)),
                                  dt_dict[n.outputs[0].dtype.name],
                                  name=n.outputs[0].name,
                                  exprs=n.params['ds_output_shape'][0])
                cn = c_out.op
                cn.name = n.name
                org_ct = emap[n.outputs[0].name]
                c_out.quantization.scales = org_ct.quantization.scales
                c_out.quantization.offsets = org_ct.quantization.offsets
                c_out.quantization.bits = org_ct.quantization.bits
                c_out.quantization.qinvariant = org_ct.quantization.qinvariant
                c_out.quantization.mins = org_ct.quantization.mins
                c_out.quantization.maxs = org_ct.quantization.maxs
                emap[n.outputs[0].name] = c_out
        else:
            cn = _Node(n.name, ot_dict[str(n.type)] if str(n.type) in ot_dict.keys() else _register_optype(n.type.name))
        for k, v in n.params.items():
            if k.startswith('ds_'):
                if n.type == OpType.Input:
                    continue
                cn.params[k] = str_to_ds_expr(v)
                continue
            try:
                cn.params[k] = cast_to_NodeParamValue(v)
            except Exception as e:
                cn.params[k] = cast_to_NodeParamValue_string(v)
        for k, v in n.constants.items():
            dtype = dt_dict[v.dtype.name]
            ct = _Tensor.new(v.to_numpy().astype(dtype.np))
            ct.dtype = dtype
            ct.quantization.scales = v.scale.flatten().cpu()
            ct.quantization.offsets = v.zerop.flatten().cpu()
            if v.qbits:
                ct.quantization.bits = v.qbits
            elif 'weights' == k:
                ct.quantization.bits = n.attrs['q_bits_weight'] if 'q_bits_weight' in n.attrs else -1
            elif 'biases' == k:
                ct.quantization.bits = n.attrs['q_bits_bias'] if 'q_bits_bias' in n.attrs else -1
            ct.quantization.qinvariant = v.qinvariant
            cn.constants[k] = ct
            if f"{k}_range" in cn.params:
                mins, maxs = cn.params[f"{k}_range"][0], cn.params[f"{k}_range"][1]
                ct.quantization.mins = mins if isinstance(mins, (list, tuple)) else [mins]
                ct.quantization.maxs = maxs if isinstance(maxs, (list, tuple)) else [maxs]
                cn.params.pop(f"{k}_range")
        for v in n.inputs:
            cn.add_input(emap[v.name])
        for v in n.outputs:
            t = emap[v.name]
            set_default_layout(t.shape)
            cn.add_output(t)

        for ak, av in n.attrs.items():
            if ak in cn.attrs:  # like workspace/.dot_section has existed when Node.create
                continue
            else:
                try:
                    cn.attrs[ak] = cast_to_NodeParamValue(av)
                except:
                    cn.attrs[ak] = av
        cg.add_node(cn)
    cg.input_tensors = _TensorList([emap[v.name] for v in g.input_tensors])
    cg.output_tensors = _TensorList([emap[v.name] for v in g.output_tensors])
    cg.topological_sort()
    cg.reset_layer_id()
    if has_ds:
        for n in cg.nodes:
            n.infer_shape()
    return cg


def parse_graph_from_ir(ir_txt, ir_bin):
    from AIPUBuilder.Optimizer.logger import OPT_INFO, OPT_WARN
    message = 'Invalid IR, please use "aipuchecker" to diagnose it for more specific information.'

    def section2str(section):
        ss = ''
        for k, v in section.items():
            ss += f'{k}={v}\n'
        ss += '\n'
        return ss

    g_str = ''
    with open(ir_txt, 'r') as ftxt:
        g_str += ftxt.read()

    rg = None
    ir_bin_map = {}
    # ir_bin_map : {ir_bin: {mmap : mmap, occupiance: 1}}

    #############################################################
    def parse_graph_from_ir_gstr(gstr, ir_bin):
        from AIPUBuilder.Optimizer.framework.qgraph import QuantizeGraph
        from AIPUBuilder.Optimizer.framework.pycore.pynode import PyNode
        from AIPUBuilder.Optimizer.framework.pycore.pytensor import PyTensor, TensorShape
        from AIPUBuilder.Optimizer.framework.pycore.pytype import register_optype, OpType
        from AIPUBuilder.Optimizer.logger import OPT_INFO, OPT_WARN, OPT_DEBUG, tqdm
        from AIPUBuilder.Optimizer.utils.dtype_utils import to_list, dtype2bits, is_float
        import mmap
        import os
        import subprocess
        import sys
        import re
        import numpy as np
        import torch
        silent_load = 'AIPUOPT_SILENTLOADING' in os.environ
        g = QuantizeGraph()
        if not silent_load:
            OPT_INFO('Suggest using "aipuchecker" to validate the IR firstly if you are not sure about its validity.')
        # get sections of key value pairs
        msg = 'Invalid IR, please use "aipuchecker" to diagnose it for more specific information.'
        try:
            sections = []
            sdict = {}
            gstr += '\n\n'
            for line in gstr.splitlines():
                if len(line.strip()) > 0:
                    k, v = line.strip().split('=')
                    sdict[k.strip()] = v.strip()
                else:
                    if len(sdict.keys()) > 0:
                        sections.append(sdict)
                    sdict = {}
            if len(sections) > 1:
                abstract = sections[0]
                g.name = abstract['model_name']
                if 'compat_quantized_model' in abstract:
                    g.compat_quantized_model = True if abstract['compat_quantized_model'].lower() == 'true' else False
                if 'dynamic_symbols' in abstract:
                    g.dynamic_symbols = cast_from_NodeParamValue_string(abstract['dynamic_symbols'])
                inp_tensor_names = cast_from_NodeParamValue_string(
                    abstract['input_tensors']) if 'input_tensors' in abstract.keys() else []
                inp_tensor_names = [str(s) for s in inp_tensor_names]
                out_tensor_names = cast_from_NodeParamValue_string(
                    abstract['output_tensors']) if 'output_tensors' in abstract.keys() else []
                out_tensor_names = [str(s) for s in out_tensor_names]
                emap = {}
                total_size = 0
                for i in range(1, len(sections)):
                    sec = sections[i]
                    top_names = cast_from_NodeParamValue_string(sec['layer_top'])
                    top_names = [str(s) for s in top_names]
                    top_shape = cast_from_NodeParamValue_string(sec['layer_top_shape'])
                    top_dtype = cast_from_NodeParamValue_string(sec['layer_top_type'])
                    top_range = cast_from_NodeParamValue_string(
                        sec['layer_top_range']) if 'layer_top_range' in sec.keys() else []
                    top_scale = cast_from_NodeParamValue_string(
                        sec['layer_top_scale']) if 'layer_top_scale' in sec.keys() else []
                    top_zerop = cast_from_NodeParamValue_string(
                        sec['layer_top_zp']) if 'layer_top_zp' in sec.keys() else []
                    for j in range(len(top_names)):
                        t = PyTensor(top_names[j])
                        t.ir_shape = TensorShape(top_shape[j])
                        total_size += np.prod(t.ir_shape)
                        t.ir_dtype = top_dtype[j]
                        t.dtype = t.ir_dtype
                        if len(top_range) > j:
                            t.ir_range = top_range[j]
                        if len(top_scale) > j:
                            t.scale = torch.tensor(top_scale[j], device=t.betensor.device) if isinstance(
                                top_scale[j], list) else top_scale[j]
                        if len(top_zerop) > j:
                            t.zerop = torch.tensor(top_zerop[j], device=t.betensor.device) if isinstance(
                                top_zerop[j], list) else top_zerop[j]
                        emap[t.name] = t
                if not silent_load:
                    OPT_INFO(f"{g.name} IR loaded.")
                tensor_list = []
                need_reordering = False
                old_offset = -1
                pbar = tqdm(range(1, len(sections)), desc="Building graph", file=sys.stdout, disable=silent_load)
                for i in pbar:
                    sec = sections[i]
                    n = PyNode(sec['layer_name'], register_optype(sec['layer_type']))
                    n.attrs['layer_id'] = sec['layer_id']
                    bottom_names = cast_from_NodeParamValue_string(sec['layer_bottom'])
                    bottom_names = [str(s) for s in bottom_names]
                    for j in range(len(bottom_names)):
                        t = emap[bottom_names[j]]
                        n.add_input(t)
                    top_names = cast_from_NodeParamValue_string(sec['layer_top'])
                    top_names = [str(s) for s in top_names]
                    for j in range(len(top_names)):
                        t = emap[top_names[j]]
                        n.add_output(t)
                    non_param_keys = ['layer_id', 'layer_name', 'layer_type', 'layer_bottom', 'layer_bottom_shape',
                                      'layer_bottom_type', 'layer_top', 'layer_top_shape', 'layer_top_type', 'layer_top_range']
                    #   'layer_top_scale', 'layer_top_zp']
                    for key in sec.keys():
                        if re.match(r'.+_offset', key):
                            ckey = key[:-7]
                            ckey_offset = key
                            ckey_type = ckey + '_type'
                            ckey_size = ckey + '_size'
                            ckey_shape = ckey + '_shape'
                            ckey_range = ckey + '_range'
                            if ckey_type in sec.keys() and ckey_size in sec.keys() and ckey_shape in sec.keys():
                                bytes_offsets = to_list(cast_from_NodeParamValue_string(sec[ckey_offset]))
                                bytes_sizes = to_list(cast_from_NodeParamValue_string(sec[ckey_size]))
                                ir_shapes = to_list(cast_from_NodeParamValue_string(sec[ckey_shape]))
                                ir_dtypes = to_list(cast_from_NodeParamValue_string(sec[ckey_type]))
                                ir_ranges = to_list(cast_from_NodeParamValue_string(
                                    sec[ckey_range])) if ckey_range in sec.keys() else []
                                ele_len = len(bytes_offsets)
                                for idx, bytes_offset in enumerate(bytes_offsets):
                                    if bytes_offset is None:
                                        OPT_WARN(f"when parser IR, {key}'s offset is None.")
                                        continue
                                    bytes_size = bytes_sizes[idx]
                                    if bytes_size == 0:
                                        OPT_WARN(f"when parser IR, {key}'s size == 0.")
                                        continue
                                    if not need_reordering and bytes_offset < old_offset:
                                        need_reordering = True
                                    old_offset = bytes_offset
                                    bytes_size = bytes_sizes[idx]
                                    t = PyTensor(f'{n.name}{ckey}')
                                    shape = ir_shapes[idx] if isinstance(ir_shapes, list) and len(
                                        ir_shapes) and isinstance(ir_shapes[idx], list) else ir_shapes
                                    t.ir_shape = TensorShape(shape)
                                    total_size += np.prod(t.ir_shape)
                                    t.ir_dtype = ir_dtypes[idx]
                                    t.dtype = t.ir_dtype
                                    if len(ir_ranges) > idx:
                                        t.ir_range = ir_ranges[idx]
                                    tensor_list.append([bytes_offset, bytes_size, t])
                                    # if layer_top_range/layer_top_scale/layer_top_zp are constants, we will put these data
                                    # to node.constants, and if multi outputs, use layer_top_range_0/layer_top_range_1
                                    # as constants key.
                                    ckey_name = f"{ckey}_{idx}" if ele_len > 1 else ckey
                                    n.constants[ckey_name] = t
                                    non_param_keys.extend([ckey_offset, ckey_type, ckey_size, ckey_shape, ckey_range])
                    for key in sec.keys():
                        if key not in non_param_keys:
                            n.params[key] = cast_from_NodeParamValue_string(sec[key])
                    g.nodes.append(n)
                pbar.refresh()

                # make sure graph can always be constructed even ir_bin is not existed
                if not os.path.exists(ir_bin):
                    OPT_WARN(f'ir_bin file "{ir_bin}" not existed when parse_graph_from_ir')
                else:
                    if not silent_load:
                        OPT_INFO("Begin to load weights.")
                    if ir_bin not in ir_bin_map:
                        f = open(ir_bin, "rb")
                        fsize = os.path.getsize(ir_bin)
                        if fsize == 0 or len(tensor_list) == 0:
                            bstr = b''
                        else:
                            bstr = mmap.mmap(f.fileno(), fsize, access=mmap.ACCESS_READ)
                            page_size = mmap.PAGESIZE
                            page_count = fsize // page_size
                            cores = torch.multiprocessing.cpu_count()  # Use multicores to preload
                            pages_per_core = page_count // cores
                            jobs = [[core * pages_per_core, (core+1) * pages_per_core] for core in range(cores)]

                            def load_page(ranges):
                                for i in range(ranges[0], ranges[1]):
                                    _ = bstr[i * 4096]
                            ps = []
                            for core in range(cores):
                                p = torch.multiprocessing.Process(target=load_page, args=(jobs[core],))
                                p.start()
                                ps.append(p)
                            for p in ps:
                                p.join()
                        ir_bin_map[ir_bin] = (f, bstr, fsize)
                    else:
                        f, bstr, fsize = ir_bin_map[ir_bin]
                    if not silent_load:
                        OPT_INFO("Weights loaded.")
                    global_offset = 0
                    forward_threshold = 2**29  # 512MB
                    if need_reordering:
                        tensor_list = sorted(tensor_list, key=lambda x: x[0])
                    pbar = tqdm(tensor_list, desc="Deserializing bin", file=sys.stdout, disable=silent_load)
                    for bytes_offset, bytes_size, t in pbar:
                        bytes_offset -= global_offset
                        tmp = PyTensor("bintmp")
                        tmp.frombuffer(bstr[bytes_offset: bytes_offset + bytes_size], dtype=t.dtype)
                        if bytes_offset + bytes_size > forward_threshold:
                            OPT_DEBUG(
                                "Due to reading large bin file, OPT will split the large file to accelerate loading process.", log_once=True)
                            global_offset += bytes_offset + bytes_size
                            global_offset -= global_offset % mmap.ALLOCATIONGRANULARITY
                            f.close()
                            f = open(ir_bin, "rb")
                            if global_offset < fsize:
                                bstr = mmap.mmap(f.fileno(), fsize - global_offset,
                                                 access=mmap.ACCESS_READ, offset=int(global_offset))
                        t.betensor = tmp.betensor.reshape(t.ir_shape)
                        if dtype2bits(t.dtype) <= 8 and is_float(t.dtype):
                            # torch.float8 only valid on cpu or specific gpu, and only valid for few operators
                            t.betensor = t.betensor.bfloat16()
                    pbar.refresh()

                inp_tensors = []
                for tname in inp_tensor_names:
                    inp_tensors.append(emap[tname])
                g.input_tensors = tuple(inp_tensors)
                out_tensors = []
                for tname in out_tensor_names:
                    out_tensors.append(emap[tname])
                g.output_tensors = tuple(out_tensors)
                g.init_networkx()
                if len(g.input_tensors) < 1:
                    OPT_WARN(
                        "There is no 'input_tensors' field in IR, will guess them according to the graph structure, which may get unexpected input_tensors.")
                    OPT_WARN("The guessed input_tensors:")
                    inp_tensors = []
                    for n in g.nodes:
                        if OpType.Input == n.type:
                            for t in n.outputs:
                                inp_tensors.append(t)
                                OPT_WARN(t.name)
                    g.input_tensors = tuple(inp_tensors)
                if len(g.output_tensors) < 1:
                    OPT_WARN(
                        "There is no 'output_tensors' field in IR, will guess them according to the graph structure, which may get unexpected output_tensors.")
                    OPT_WARN("The guessed output_tensors:")
                    out_tensors = []
                    for n in g.nodes:
                        if len(n.children) < 1:
                            for t in n.outputs:
                                out_tensors.append(t)
                                OPT_WARN(t.name)
                    g.output_tensors = tuple(out_tensors)

                # compatiable the constant scale zp in constant data
                opt_mini_keys = {'scale': 'scale', 'zp': 'zerop'}
                mini_keys = ['scale', 'zp', 'range']
                for n in g.nodes:
                    need_pop_key = []
                    o_len = len(n.outputs)
                    for oi, ot in enumerate(n.outputs):
                        for mini_key in mini_keys:
                            c_full_key = f"layer_top_{mini_key}_{oi}" if o_len > 1 else f"layer_top_{mini_key}"
                            p_full_key = f"layer_top_{mini_key}"
                            if c_full_key in n.constants:
                                if mini_key != 'range':  # layer_top_scale/layer_top_zp in constants
                                    ot.__setattr__(opt_mini_keys[mini_key], n.constants[c_full_key].betensor)
                                    ot.key_axis = n.params['activation_quantization_axis'][oi]
                                else:
                                    ot.ir_range = n.constants[c_full_key].betensor.cpu().tolist()
                                need_pop_key.append(c_full_key)
                            elif p_full_key in n.params:
                                if mini_key != 'range':
                                    ot.__setattr__(opt_mini_keys[mini_key], n.params[p_full_key][oi])
                                    need_pop_key.append(p_full_key)
                    for k, t in n.constants.items():
                        for mini_key in mini_keys:
                            if k.endswith(f"_{mini_key}") and k not in need_pop_key:
                                main_key = k[0:-len(f"_{mini_key}")]
                                if main_key not in n.constants or mini_key == 'range':
                                    if mini_key != 'range':
                                        n.params[k] = t.to_numpy().flatten().tolist()
                                    elif mini_key not in n.params:
                                        n.params[k] = t.betensor.cpu().tolist()
                                else:
                                    if n.constants[k].betensor.numel() > 0:
                                        n.constants[main_key].__setattr__(
                                            opt_mini_keys[mini_key], n.constants[k].betensor)
                                need_pop_key.append(k)
                            elif f"{k}_{mini_key}" in n.params and mini_key != 'range':
                                n.constants[k].__setattr__(opt_mini_keys[mini_key], n.params[f"{k}_{mini_key}"])
                                # delete the weights_scale/zerop, otherwise qat model in optforward will reread these params
                                need_pop_key.append(f"{k}_{mini_key}")
                    for pop_key in set(need_pop_key):
                        if pop_key in n.constants:
                            n.constants.pop(pop_key)
                        elif pop_key in n.params:
                            n.params.pop(pop_key)
                        else:
                            OPT_WARN(f"pop_key = {pop_key} is not in {n} constants or params.")
                if not silent_load:
                    OPT_INFO('Successfully parsed IR with python API.')
        except Exception as e:
            OPT_WARN(f'Failed to parse IR with the exception msg: {e}')
            OPT_WARN(msg)
            raise e

        return g

    #############################################################
    try:
        sections = []
        sdict = {}
        g_str += '\n\n'
        for line in g_str.splitlines():
            if len(line.strip()) > 0:
                k, v = line.strip().split('=')
                sdict[k.strip()] = v.strip()
            else:
                if len(sdict.keys()) > 0:
                    sections.append(sdict)
                sdict = {}
        sgstr_map = {}
        mgstr = ''
        i = 0
        while i < len(sections):
            sec = sections[i]
            if 'subgraph_name' in sec.keys() and 'input_tensors' in sec.keys():
                # subgraph part
                sg_name = sec['subgraph_name']
                sec['model_name'] = sg_name
                sgstr_map[sg_name] = ''
                sg_layers = int(sec['layer_number'])
                for j in range(sg_layers+1):
                    sgstr_map[sg_name] += section2str(sections[i+j])
                i += sg_layers + 1
            else:
                mgstr += section2str(sec)
                i += 1
        # assume root graph and all subgraphs are all DAG
        rg = parse_graph_from_ir_gstr(mgstr, ir_bin)
        for sg_name, sgstr in sgstr_map.items():
            rg.subgraph_map[sg_name] = parse_graph_from_ir_gstr(sgstr, ir_bin)
            rg.subgraph_map[sg_name].root_graph = rg
        for loaded_bin in list(ir_bin_map.keys()):
            f, bstr, fsize = ir_bin_map[loaded_bin]
            f.close()
            del bstr
            del ir_bin_map[loaded_bin]
    except Exception as e:
        OPT_WARN(f'Failed to parse IR with the exception msg: {e}')
        OPT_WARN(message)
        raise e

    return rg


def serialize_graph_to_ir(g, ir_txt, ir_bin):
    from AIPUBuilder.Optimizer.utils import (dtype2str, dtype2range, dtype2bytes, make_path, torch_type2dtype,
                                             is_torch_tensor_with_multi_data, is_torch_tensor, is_float, dtype2bits)
    from AIPUBuilder.Optimizer.logger import OPT_INFO, OPT_ERROR, OPT_WARN, tqdm
    from AIPUBuilder.Optimizer.framework import PyTensor, Dtype
    import mmap
    import threading
    import sys

    def _convert_scale_zp_to_list(data):
        import torch
        import numpy as np
        ret_data = None
        if isinstance(data, (float, int)):
            ret_data = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            ret_data = data.tolist()
            ret_data = _convert_scale_zp_to_list(ret_data)
        else:
            ret_data = data
        return ret_data

    gstr = ''
    gstr += f'model_name={g.name}\nlayer_number={len(g.nodes)}\n'
    inp_tensor_names = []
    for t in g.input_tensors:
        inp_tensor_names.append(t.name)
    gstr += f'input_tensors={cast_to_NodeParamValue_string(inp_tensor_names)}\n'
    out_tensor_names = []
    for t in g.output_tensors:
        out_tensor_names.append(t.name)
    gstr += f'output_tensors={cast_to_NodeParamValue_string(out_tensor_names)}\n'
    gstr += '\n'
    offset = 0
    tensor_list = []
    separate_points = []
    all_nodes = [] + g.nodes
    all_nodes_num = len(g.nodes)
    for _, sg in g.subgraph_map.items():
        all_nodes += sg.nodes
        separate_points.append((all_nodes_num, sg))
        all_nodes_num += len(sg.nodes)
    sp_idx = 0
    lid = 0
    pbar = tqdm(enumerate(all_nodes), desc="Writing IR", file=sys.stdout)
    for i, n in pbar:
        if (sp_idx < len(separate_points)) and (i == separate_points[sp_idx][0]):
            sg = separate_points[sp_idx][1]
            gstr += '\n'
            gstr += f'subgraph_name={sg.name}\nlayer_number={len(sg.nodes)}\n'
            inp_tensor_names = []
            for t in sg.input_tensors:
                inp_tensor_names.append(t.name)
            gstr += f'input_tensors={cast_to_NodeParamValue_string(inp_tensor_names)}\n'
            out_tensor_names = []
            for t in sg.output_tensors:
                out_tensor_names.append(t.name)
            gstr += f'output_tensors={cast_to_NodeParamValue_string(out_tensor_names)}\n'
            gstr += '\n'
            sp_idx += 1
            lid = 0
        gstr += f'layer_id={lid}\nlayer_name={n.name}\nlayer_type={n.type.name}\n'
        lid += 1
        bottom_names = []
        bottom_shape = []
        bottom_dtype = []
        for t in n.inputs:
            bottom_names.append(t.name)
            bottom_shape.append(list(t.ir_shape))
            bottom_dtype.append(dtype2str(t.dtype))
        top_names = []
        top_shape = []
        top_dtype = []
        top_scale = []
        top_zerop = []
        top_range = []
        top_key_axis = []
        for t in n.outputs:
            top_names.append(t.name)
            top_shape.append(list(t.ir_shape))
            top_dtype.append(dtype2str(t.dtype))
            top_scale.append(t.scale)
            top_zerop.append(t.zerop)
            if (t.qmin is None or t.qmax is None) and not n.unquantifiable:
                t.qmin, t.qmax = dtype2range(t.dtype)
            top_range.append([t.qmin, t.qmax])
            top_key_axis.append(t.key_axis)
        gstr += f'layer_bottom={cast_to_NodeParamValue_string(bottom_names)}\n'
        gstr += f'layer_bottom_shape={cast_to_NodeParamValue_string(bottom_shape)}\n'
        gstr += f'layer_bottom_type={cast_to_NodeParamValue_string(bottom_dtype)}\n'
        gstr += f'layer_top={cast_to_NodeParamValue_string(top_names)}\n'
        gstr += f'layer_top_shape={cast_to_NodeParamValue_string(top_shape)}\n'
        gstr += f'layer_top_type={cast_to_NodeParamValue_string(top_dtype)}\n'
        if not n.unquantifiable:
            gstr += f'layer_top_range={cast_to_NodeParamValue_string(top_range)}\n'
        if any([is_torch_tensor_with_multi_data(s) for s in top_scale]):
            scale_type, scale_offset, scale_size, scale_shape = [], [], [], []
            zp_type, zp_offset, zp_size, zp_shape = [], [], [], []
            for ot in n.outputs:
                scale = ot.scale
                zp = ot.zerop
                scale_mem_size = scale.element_size() * scale.numel()
                zp_mem_size = zp.element_size() * zp.numel()
                scale_type.append(dtype2str(torch_type2dtype(scale.dtype)))
                scale_offset.append(offset)
                scale_shape.append(list(scale.shape))
                scale_size.append(scale_mem_size)
                tensor_list.append([offset, scale_mem_size, PyTensor(
                    'tmp_scale', scale, dtype=torch_type2dtype(scale.dtype))])
                offset += scale_mem_size
                zp_type.append(dtype2str(torch_type2dtype(zp.dtype)))
                zp_offset.append(offset)
                zp_shape.append(list(zp.shape))
                zp_size.append(zp_mem_size)
                tensor_list.append([offset, zp_mem_size, PyTensor('temp_zp', zp, dtype=torch_type2dtype(zp.dtype))])
                offset += zp_mem_size
            gstr += f"layer_top_scale_type=[{','.join(scale_type)}]\n"
            gstr += f"layer_top_scale_offset={scale_offset}\n"
            gstr += f"layer_top_scale_shape={scale_shape}\n"
            gstr += f"layer_top_scale_size={scale_size}\n"
            gstr += f"layer_top_zp_type=[{','.join(zp_type)}]\n"
            gstr += f"layer_top_zp_offset={zp_offset}\n"
            gstr += f"layer_top_zp_shape={zp_shape}\n"
            gstr += f"layer_top_zp_size={zp_size}\n"
        else:
            top_scale = [ts.tolist()[0] for ts in top_scale]
            top_zerop = [tz.tolist()[0] for tz in top_zerop]
            gstr += f'layer_top_scale={cast_to_NodeParamValue_string(top_scale)}\n'
            gstr += f'layer_top_zp={cast_to_NodeParamValue_string(top_zerop)}\n'

        """
        move the scale/zerop of constant tensor to constant, in order to dequantize the constant
        to float using its scale/zerop in the subsequent process.
        """
        def move_constant_tensor_scale_to_constant(node, key):
            if key in node.constants:
                scale = node.constants[key].scale
                zerop = node.constants[key].zerop
                s_key, z_key = f"{key}_scale", f"{key}_zp"
                if is_torch_tensor_with_multi_data(scale):
                    node.constants[s_key] = PyTensor(s_key, scale)
                    node.constants[z_key] = PyTensor(z_key, zerop)
                    # if weight is fp4/fp8 quantization, lib requires the zp's dtype == weights's dtype
                    # so change the n.constants['weights_zp'].dtype from bfloat16 to weight's dtype
                    # and the n.constants['weights_zp'] data dtype(now still is bfloat16) will
                    # change at pytensor.tobytes() when serializing bin
                    if key == "weights":
                        weight_dtype = node.constants["weights"].dtype
                        if is_float(weight_dtype) and (dtype2bits(weight_dtype)) <= 8:
                            node.constants[z_key].dtype = weight_dtype
                elif is_torch_tensor(scale) and scale.numel() == 1:
                    node.params[s_key] = scale.item()
                    node.params[z_key] = zerop.item()
                else:
                    pass

        """
        now only put the scale/zerop of weights and biases to ir, if need more scale/zp of constant data to ir, please
        add to constant_keys variable.
        """
        constant_keys = ['weights', 'biases']
        for ck in constant_keys:
            move_constant_tensor_scale_to_constant(n, ck)

        for c in n.constants.keys():
            ct = n.constants[c]
            c_size = dtype2bytes(ct.dtype) * ct.betensor.numel()
            gstr += f'{c}_type={dtype2str(ct.dtype)}\n'
            gstr += f'{c}_offset={offset}\n'
            gstr += f'{c}_size={c_size}\n'
            gstr += f'{c}_shape={cast_to_NodeParamValue_string(list(ct.betensor.shape))}\n'
            ct_value = ct
            min_v, max_v = dtype2range(ct.dtype)
            if not is_float(ct.dtype) and (ct_value.betensor.max() > max_v or ct_value.betensor.min() < min_v):
                OPT_WARN(
                    f"Node: {n}, constant: {c} : data overflow, with min/max {ct_value.betensor.min()}/{ct_value.betensor.max()} under dtype {ct.dtype}")
            tensor_list.append([offset, c_size, ct_value])
            offset += c_size

        n.params['activation_quantization_axis'] = top_key_axis
        for k, v in n.params.items():
            gstr += f'{k}={cast_to_NodeParamValue_string(v)}\n'
        gstr += '\n'
    pbar.refresh()
    make_path(ir_txt)
    make_path(ir_bin)
    with open(ir_txt, 'w') as ftxt:
        ftxt.write(gstr)

    if len(tensor_list) == 0:
        with open(ir_bin, 'wb') as fbin:
            pass
        return

    total_size = tensor_list[-1][0] + tensor_list[-1][1]
    mem_seg = 2 ** 29 - (2**29 % mmap.ALLOCATIONGRANULARITY)  # 512MB
    for offset, size, ct in tensor_list:
        size = size - (size % mmap.ALLOCATIONGRANULARITY)
        mem_seg = max(mem_seg, size)

    job_list = []
    job = []
    workloads = 0
    for offset, size, ct in tensor_list:
        offset = offset % mem_seg
        job.append([offset, size, ct])
        if offset + size > mem_seg:
            remains = mem_seg - offset
            job_list.append(job)
            workloads += len(job)
            job = [[0, size - remains, ct]]
    if len(job) > 0:
        job_list.append(job)
        workloads += len(job)

    def write_to_kernel(origin_offset, job, fileno, pbar):
        length = mem_seg
        if origin_offset + mem_seg > total_size:
            length = total_size - origin_offset
        mem = mmap.mmap(fileno, length, flags=mmap.MAP_SHARED, access=mmap.ACCESS_WRITE, offset=origin_offset)
        for offset, size, ct in job:
            bytes = ct.tobytes()
            if size == len(bytes):
                if offset + size <= mem_seg:
                    mem[offset:offset+size] = bytes
                else:
                    mem[offset:] = bytes[:length-offset]
            else:
                mem[:size] = bytes[-size:]
            pbar.update(1)
        mem.close()

    cores = len(job_list)
    # alloc space on disk
    fbin = open(ir_bin, 'wb')
    fbin.seek(total_size-1)
    fbin.write(b'\0')
    fbin.close()
    fbin = open(ir_bin, 'r+b')

    tpbar = tqdm(total=workloads, desc="Serializing bin", file=sys.stdout)
    ps = []
    for i in range(cores):
        p = threading.Thread(target=write_to_kernel, args=(i*mem_seg, job_list[i], fbin.fileno(), tpbar))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()
    tpbar.refresh()

    fbin.close()
