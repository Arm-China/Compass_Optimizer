# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import OPT_INFO
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.utils.quant_tool_utils import ASYM2SYM_OP_DICT, QuantMode, OP_NEED_ALIGN_INP_OUT_DTYPE
import re
import torch

__all__ = [
    "InsertCastOp",
    "InsertQuantizeOp",
    "InsertDeQuantizeOp",
    "InsertPadOp"
]


class BaseInsertOp(object):
    def __init__(self, g):
        self.g = g

    def criteria(self):
        raise NotImplementedError()

    def insert(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()


class InsertCastOp(BaseInsertOp):
    def __init__(self, g, config):
        super().__init__(g)
        self.cast_dtypes_for_lib = config.cast_dtypes_for_lib
        self.op_need_cast_dtypes_for_lib = set()
        self.inserted_cast_layers = []

    def whether_an_inserted_cast_layer(self, node):
        return node.type == OpType.Cast and node.additional

    def whether_need_adapt_input_dtype(self, node, parent_node, edge_tensor):
        if node.type not in self.op_need_cast_dtypes_for_lib:
            return False
        inputs_dtype = []
        outputs_type = []
        for tidx, t in enumerate(node.inputs):
            inputs_dtype.append(t.dtype)
        for tidx, t in enumerate(node.outputs):
            outputs_type.append(t.dtype)
        dtype_spec = get_op_dtype_spec(node.type)
        is_spec_valid = True
        if len(dtype_spec) > 0:
            if len(inputs_dtype) != len(dtype_spec[0].in_dtypes):
                is_spec_valid = False
        else:
            is_spec_valid = False
        matched = False
        for spec in dtype_spec:
            spec_in_dtypes = spec.in_dtypes
            spec_in_dtypes_num = len(spec_in_dtypes)
            matched_this = True
            for tidx, dt in enumerate(inputs_dtype):
                if tidx >= spec_in_dtypes_num:
                    matched_this = False
                    break
                if spec_in_dtypes[tidx] != dt:
                    matched_this = False
                    break
            if matched_this:
                # whether match output
                if outputs_type == spec.out_dtypes:
                    matched = True
                    break
                else:
                    continue
        return is_spec_valid and (not matched)

    def criteria(self):
        _conditions = [
            # insert cast op for mixed precision quantization
            lambda node, parent_node, edge_tensor:
            (parent_node.attrs['q_bits_activation'] != node.attrs['q_bits_activation'])
            and (not self.whether_an_inserted_cast_layer(parent_node))
            and (not self.whether_an_inserted_cast_layer(node))
            and (not node.get_param('unquantifiable', optional=True, default_value=False))
            and (not parent_node.get_param('unquantifiable', optional=True, default_value=False)),
            # insert cast op for lib's dtypes spec
            lambda node, parent_node, edge_tensor:
            (node.type in self.op_need_cast_dtypes_for_lib)
            and (not self.whether_an_inserted_cast_layer(parent_node))
            and self.whether_need_adapt_input_dtype(node, parent_node, edge_tensor)
            and (not node.get_param('unquantifiable', optional=True, default_value=False)),
            # insert cast op for OPs like lstm, gru which ask for specific inputs be quantized with symmetric mode
            lambda node, parent_node, edge_tensor:
            (node.type in ASYM2SYM_OP_DICT)
            and (not self.whether_an_inserted_cast_layer(parent_node))
            and (not node.get_param('unquantifiable', optional=True, default_value=False)),
        ]
        return _conditions

    def insert(self):
        _conditions = self.criteria()

        for _cond in _conditions:
            self.inserted_cast_layers += self.g.insert_cast_op_ahead(condition_func=_cond)

        return self.inserted_cast_layers

    def before_pass(self):
        # parse ops need cast dtypes to align with lib's spec
        cast_dtypes_for_lib = str(self.cast_dtypes_for_lib).lower().strip()
        if 'false' != cast_dtypes_for_lib:
            if 'true' == cast_dtypes_for_lib:
                for node in self.g.nodes:
                    self.op_need_cast_dtypes_for_lib.add(node.type)
            else:  # op list
                op_list = [x.lower().strip() for x in re.split(r',|\s+', cast_dtypes_for_lib) if x.lower().strip()]
                for node in self.g.nodes:
                    if str(node.type)[7:].lower() in op_list:
                        self.op_need_cast_dtypes_for_lib.add(node.type)

        if len(self.op_need_cast_dtypes_for_lib) > 0:
            OPT_INFO("These OPs will automatically cast dtypes to adapt to lib's dtypes' spec (may cause model accuracy loss due to corresponding spec's restriction): ")
            OPT_INFO(str(self.op_need_cast_dtypes_for_lib))
        self.op_need_cast_dtypes_for_lib.discard(OpType.Cast)

    def after_pass(self):
        # adaptor_cast_to_lib_implementation(inserted_cast_layers):
        def set_cast_totype(node):
            def mix_quantization_condition(
                node): return node.parents[0].attrs['q_bits_activation'] != node.children[0].attrs['q_bits_activation']
            output_dtype = []
            for output in node.children[0].outputs:
                if mix_quantization_condition(node) and False == output.qinvariant:
                    output_dtype.append(bits2dtype(
                        node.children[0].attrs['q_bits_activation'], is_signed(output.dtype)))
                else:
                    output_dtype.append(output.dtype)
            dtype_spec = get_op_dtype_spec(node.children[0].type)
            candidates = []
            backups = []
            for spec in dtype_spec:
                spec_output_type = spec.out_dtypes
                dt_int = True
                for dt in spec.out_dtypes + spec.in_dtypes:
                    if is_float(dt):
                        dt_int = False
                        break
                if not dt_int:
                    continue
                if output_dtype == spec_output_type:
                    candidates.append([spec.in_dtypes, 0.0])
                backups.append([spec.in_dtypes, 0.0])
            if len(candidates) < 1:
                candidates = backups
            for i in range(len(candidates)):
                score = 0.0
                for j, in_dtype in enumerate(candidates[i][0]):
                    if j > len(node.children[0].inputs) - 1:
                        continue
                    if mix_quantization_condition(node) and False == node.children[0].inputs[j].qinvariant:
                        dt = bits2dtype(node.children[0].attrs['q_bits_activation'],
                                        is_signed(node.children[0].inputs[j].dtype))
                    else:
                        dt = node.children[0].inputs[j].dtype
                        for cp in node.children[0].parents:
                            if cp != node and cp.type == OpType.Cast and cp.additional and cp.outputs[0] == node.children[0].inputs[j]:
                                dt = cp.params['to_dtype']
                                break

                    if is_signed(dt) == is_signed(in_dtype):
                        score += 0.01
                    if dt == in_dtype:
                        score += 2.0
                    elif dtype2bits(dt) < dtype2bits(in_dtype):
                        score += 1.0
                    elif dtype2bits(dt) == dtype2bits(in_dtype):
                        score += 1.5
                    else:
                        score += dtype2bits(in_dtype) * 1.0 / dtype2bits(dt)
                candidates[i][1] = score
            if len(candidates) > 0:
                best = sorted(candidates, key=lambda x: x[1])[-1][0]
                return best
            else:
                return []
        unique_flags = {}
        for n in self.inserted_cast_layers:
            if (n.children[0].type in self.op_need_cast_dtypes_for_lib) and (n.name not in unique_flags):
                unique_flags[n.name] = True
                idx = 0
                for tidx, t in enumerate(n.children[0].inputs):
                    if n.outputs[0].name == t.name:
                        idx = tidx
                        break
                cast_totype_list = set_cast_totype(n)
                if len(cast_totype_list) > idx:
                    n.params['to_dtype'] = cast_totype_list[idx]
                    if n.params['to_dtype'] != n.inputs[0].dtype:
                        OPT_INFO("layer_id=%s, %s, '%s', cast its output '%s' dtype from %s to %s due to lib's %s spec by insert a cast layer." % (str(n.parents[0].attrs['layer_id']), str(
                            n.parents[0].type), str(n.parents[0].name), str(n.inputs[0].name), str(n.inputs[0].dtype), str(n.params['to_dtype']), str(n.children[0].type)))

        for n in self.g.nodes:
            if n.type in ASYM2SYM_OP_DICT:
                for idx in range(len(n.inputs)):
                    parent = n.parents[idx]
                    if idx in ASYM2SYM_OP_DICT[n.type][0] and QuantMode.is_asymmetric(parent.attrs["q_mode_activation"]):
                        parent.attrs["q_mode_activation"] = ASYM2SYM_OP_DICT[n.type][1]

    def run(self):
        self.before_pass()
        self.insert()
        self.after_pass()
        self.inserted_cast_layers = []
        # update tensors' quantization info becasue some layers' dtype may change
        # after inserting cast and more cast layers will be needed
        self.g.set_tensor_quantization_attrs()
        self.insert()
        self.after_pass()
        self.inserted_cast_layers = []


class InsertQuantizeOp(BaseInsertOp):
    def __init__(self, graph, config):
        super().__init__(graph)
        self.config = config
        self.inserted_nodes = []

    def criteria(self):
        _condition = [
            lambda node, parent_node, edge_tensor:
            (parent_node.params['unquantifiable'] == True and node.params['unquantifiable'] == False)
        ]
        # qinvariant tensors with scale=1.0, zp=0 are safe to pass through
        return _condition

    def after_insert(self):
        for n in self.inserted_nodes:
            inpt = n.inputs[0]
            if inpt.qbits is None or inpt.qmin is None or inpt.qmax is None:
                OPT_ERROR(f"the output({inpt.name}) doesn't quantize in InsertQuantizeOp.")
            if isinstance(inpt.scale, torch.Tensor) and inpt.scale.numel() > 1:
                scale_t = PyTensor(self.g.get_valid_tensor_name(inpt.name),
                                   TensorShape(*list(inpt.scale.shape)), Dtype.FP32)
                scale_t.betensor = inpt.scale
                n.constants['quantize_scale'] = scale_t
                zerop_t = PyTensor(self.g.get_valid_tensor_name(inpt.name),
                                   TensorShape(*list(inpt.zerop.shape)), Dtype.INT32)
                zerop_t.betensor = inpt.zerop.to(torch.int32)
                n.constants['quantize_zp'] = zerop_t
            else:
                n.params['quantize_scale'] = float(inpt.scale)
                n.params['quantize_zp'] = int(inpt.zerop)

            n.params['unquantifiable'] = True
            n.outputs[0].dtype = inpt.dtype
            n.attrs['qinfo'] = {
                'qbits': inpt.qbits,
                'qmin': inpt.qmin,
                'qmax': inpt.qmax,
                'qinvariant': inpt.qinvariant,
                'dtype': inpt.dtype,
            }

    def insert(self):
        conditions = self.criteria()
        for cond_func in conditions:
            self.inserted_nodes += self.g.insert_dummy_node_ahead(OpType.Quantize, cond_func, )
        return self.inserted_nodes

    def run(self):
        self.insert()
        self.after_insert()


class InsertDeQuantizeOp(BaseInsertOp):
    def __init__(self, graph, config):
        super().__init__(graph)
        self.config = config
        self.inserted_nodes = []

    def criteria(self):
        _condition = [
            lambda node, parent_node, edge_tensor:
            (parent_node.params['unquantifiable'] == False and node.params['unquantifiable'] == True)
        ]
        # qinvariant tensors with scale=1.0, zp=0 are safe to pass through
        return _condition

    def after_insert(self):
        for n in self.inserted_nodes:
            n.params['unquantifiable'] = True
            inpt = n.inputs[0]
            if isinstance(inpt.scale, torch.Tensor) and inpt.scale.numel() > 1:
                scale_t = PyTensor(self.g.get_valid_tensor_name(inpt.name),
                                   TensorShape(*list(inpt.scale.shape)), Dtype.FP32)
                scale_t.betensor = inpt.scale
                n.constants['quantize_scale'] = scale_t
                zerop_t = PyTensor(self.g.get_valid_tensor_name(inpt.name),
                                   TensorShape(*list(inpt.zerop.shape)), Dtype.INT32)
                zerop_t.betensor = inpt.zerop.to(torch.int32)
                n.constants['quantize_zp'] = zerop_t
            else:
                n.params['quantize_scale'] = float(inpt.scale)
                n.params['quantize_zp'] = int(inpt.zerop)

    def insert(self):
        conditions = self.criteria()
        for cond_func in conditions:
            self.inserted_nodes += self.g.insert_dummy_node_ahead(OpType.DeQuantize, cond_func)
        return self.inserted_nodes

    def run(self):
        self.insert()
        self.after_insert()


class InsertPadOp(BaseInsertOp):
    def __init__(self, graph, config):
        super().__init__(graph)
        self.config = config
        self.inserted_nodes = []

    def criteria(self):
        _condition = [
            lambda node, parent_node, edge_tensor: node.type in OP_NEED_ADD_PAD_AVOID_ASYNC_DIVIDE and
            node.get_param('ceil_mode', optional=True,
                           default_value=False) == True and
            node.get_param('count_include_pad', optional=True,
                           default_value=False) == True and
            node.get_param('method') == 'AVG' and node.outputs[0].zerop != 0
        ]
        return _condition

    def insert(self):
        conditions = self.criteria()
        for cond_func in conditions:
            self.inserted_nodes += self.g.insert_pad_op_ahead(cond_func)
        return self.inserted_nodes

    def run(self):
        self.insert()
