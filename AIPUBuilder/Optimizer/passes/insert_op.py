# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import OpType, Dtype, PyNode, PyTensor  # noqa
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
    def __init__(self, graph, config=None):
        self.g = graph
        self.config = config
        self.inserted_ops = []

    def __call__(self):
        raise NotImplementedError()

    def insert_op_at_edge(self, n, parent_node, inp_t, ntype):
        _nname = parent_node.name + ("_%s_" % (str(ntype)[7:],)) + timestamp_string()
        dummy_op = PyNode(self.g.get_valid_node_name(_nname), ntype)
        dummy_op.additional = True
        dummy_op.add_input(inp_t)
        atensor_name = self.g.get_valid_tensor_name(inp_t.name + ("_%s_tensor_" % (str(ntype)[7:],)))
        atensor = inp_t.clone(atensor_name)
        dummy_op.add_output(atensor)
        idx = n.remove_input(inp_t)
        n.add_input(atensor, idx)
        dummy_op.attrs.update(parent_node.attrs.clone())
        self.g.add_node(dummy_op)
        return dummy_op

    def whether_an_inserted_op(self, node, target_type):
        return node.type == target_type and node.additional

    @staticmethod
    def in_tensor_consumers(n, t):
        consumers = []
        t_producers = t.pnode
        if t.pnode is not None:
            for pc in t_producers.children:
                if t in pc.inputs:
                    consumers.append(pc)
        return consumers


class BaseQDQOp(BaseInsertOp):
    def __init__(self, graph, config=None):
        super().__init__(graph, config)

    def _multi_branch_merge_critria(self, n, target_type):
        flag = False
        if n.type == target_type:
            consumers = InsertQuantizeOp.in_tensor_consumers(n, n.inputs[0])
            if len(consumers) > 1:
                flag = True
                for consumer in consumers:
                    if consumer.type != target_type:
                        flag = False
                        break
                    if not consumer.outputs[0].is_qinfo_equal(n.outputs[0]):
                        flag = False
                        break
                if flag:
                    return flag, consumers
        return flag, []

    def _multi_branch_merge(self, merged_nodes, reserved_n):
        for mn in merged_nodes:
            children = mn.children
            for cn in children:
                idx = cn.remove_input(mn.outputs[0])
                cn.add_input(reserved_n.outputs[0], idx)
        for mn in merged_nodes:
            self.g.remove_node(mn)

    def merge_multi_branch(self, target_type):
        for n in self.g.nodes:
            matched, merged_node = self._multi_branch_merge_critria(n, target_type)
            if matched and len(merged_node):
                merged_node.remove(n)
                self._multi_branch_merge(merged_node, n)


class InsertQuantizeOp(BaseQDQOp):
    def __init__(self, graph, config=None):
        super().__init__(graph, config)

    def __call__(self):
        # qinvariant tensors with scale=1.0, zp=0 are safe to pass through
        _conditions = [
            lambda node, parent_node, edge_tensor:
            (parent_node.get_param('unquantifiable', optional=True, default_value=False) and not edge_tensor.qinvariant
             and node.get_param('unquantifiable', optional=True, default_value=False) == False)
        ]

        for cond_func in _conditions:
            self.inserted_ops += self.g.insert_dummy_node_ahead(OpType.Quantize, cond_func, )

        for n in self.inserted_ops:
            inpt = n.inputs[0]
            if inpt.qbits is None or inpt.qmin is None or inpt.qmax is None:
                OPT_ERROR(f"the input({inpt.name}) of {n} doesn't quantize in InsertQuantizeOp.")
            n.params['unquantifiable'] = True
            n.params['round_mode'] = "ROUND_TO_EVEN"

        #  merge multi-quantize with same tensors
        self.merge_multi_branch(OpType.Quantize)


class InsertDeQuantizeOp(BaseQDQOp):
    def __init__(self, graph, config=None):
        super().__init__(graph, config)

    def __call__(self):
        _conditions = [
            lambda node, parent_node, edge_tensor: (
                not parent_node.get_param('unquantifiable', optional=True, default_value=True) and
                not edge_tensor.qinvariant and
                node.get_param('unquantifiable', optional=True, default_value=False) == True)
        ]

        for cond_func in _conditions:
            self.inserted_ops += self.g.insert_dummy_node_ahead(OpType.DeQuantize, cond_func)

        for n in self.inserted_ops:
            n.params['unquantifiable'] = True
            n.attrs['trigger_float_op'] = n.children[0].attrs['trigger_float_op']

        #  merge multi-dequantize with same tensors
        self.merge_multi_branch(OpType.DeQuantize)


class InsertPadOp(BaseInsertOp):
    def __init__(self, graph, config=None):
        super().__init__(graph, config)

    def _insert_pad_op_ahead(self, condition_func=lambda node, parent_node, edge_tensor: False):  # for avgpool cnt=ceil=true

        inserted_op_list = self.g.insert_dummy_node_ahead(OpType.Pad, condition_func)

        for n in inserted_op_list:
            n.params['mode'] = 'CONSTANT'
            n.params['constant_value'] = 0 - n.children[0].inputs[0].zerop
            pool_params = n.children[0].params
            if n.children[0].type == OpType.Pooling:
                n.params['pads'] = [[0, 0], [pool_params['pad_top'], pool_params['pad_bottom']],
                                    [pool_params['pad_left'], pool_params['pad_right']], [0, 0]]
                n.children[0].params['pad_left'] = 0
                n.children[0].params['pad_right'] = 0
                n.children[0].params['pad_top'] = 0
                n.children[0].params['pad_bottom'] = 0
            if n.children[0].type == OpType.Pooling3D:  # NDHWC
                n.params['pads'] = [[0, 0], [pool_params['pad_z_begin'], pool_params['pad_z_end']],
                                    [pool_params['pad_y_begin'], pool_params['pad_y_end']],
                                    [pool_params['pad_x_begin'], pool_params['pad_x_end']], [0, 0]]
                n.children[0].params['pad_x_begin'] = 0
                n.children[0].params['pad_x_end'] = 0
                n.children[0].params['pad_y_begin'] = 0
                n.children[0].params['pad_y_end'] = 0
                n.children[0].params['pad_z_begin'] = 0
                n.children[0].params['pad_z_end'] = 0
        return inserted_op_list

    def __call__(self):
        _conditions = [
            lambda node, parent_node, edge_tensor: node.type in OP_NEED_ADD_PAD_AVOID_ASYNC_DIVIDE and
            node.get_param('ceil_mode', optional=True, default_value=False) == True and
            node.get_param('count_include_pad', optional=True, default_value=False) == True and
            node.get_param('method') == 'AVG' and node.outputs[0].zerop != 0
        ]

        for cond_func in _conditions:
            self.inserted_ops += self._insert_pad_op_ahead(cond_func)


class InsertCastOp(BaseInsertOp):
    def __init__(self, graph, config=None):
        super().__init__(graph, config)
        self.cast_dtypes_for_lib = config.cast_dtypes_for_lib if config is not None and hasattr(
            config, 'cast_dtypes_for_lib') else False

        self.op_need_cast_dtypes_for_lib = set()
        if isinstance(self.cast_dtypes_for_lib, bool):
            self.op_need_cast_dtypes_for_lib = set([n.type for n in self.g.nodes]
                                                   ) if self.cast_dtypes_for_lib else set()
        else:
            for n in self.g.nodes:
                if self.cast_dtypes_for_lib.get(n):
                    self.op_need_cast_dtypes_for_lib.add(n.type)

        if len(self.op_need_cast_dtypes_for_lib) > 0:
            OPT_INFO(f"These OPs will automatically cast dtypes to adapt to lib's dtypes' spec "
                     f"(may cause model accuracy loss due to corresponding spec's restriction): "
                     f"{str(self.op_need_cast_dtypes_for_lib)}")
            self.op_need_cast_dtypes_for_lib.discard(OpType.Cast)

    def whether_an_inserted_op(self, node, target_type=OpType.Cast):
        return super().whether_an_inserted_op(node, target_type)

    def insert_cast_at_edge(self, n, parent_node, inp_t, ntype=OpType.Cast):
        dummy_op = self.insert_op_at_edge(n, parent_node, inp_t, ntype)
        dummy_op.params['only_for_quantized'] = True
        dummy_op.params['to_dtype'] = bits2dtype(
            dummy_op.attrs['q_bits_activation'], is_signed=is_signed(dummy_op.outputs[0].dtype))
        dummy_op.params['ignore_scale_zp'] = True
        dummy_op.params['clip_mode'] = 'TRUNCATION'
        return dummy_op

    def _whether_need_adapt_input_dtype(self, node):
        if node.type not in self.op_need_cast_dtypes_for_lib:
            return False
        inputs_dtype = [t.dtype for t in node.inputs]
        outputs_dtype = [t.dtype for t in node.outputs]
        lib_dtype_spec = node.get_lib_dtype_spec()
        lib_not_impl = False
        if len(lib_dtype_spec) > 0:
            matched = False
            for spec in lib_dtype_spec:
                # check the inputs dtype and outputs dtype is matched with the lib spec
                indtype_matched = spec.in_dtypes == inputs_dtype  # whether matcn input dtypes
                matched = indtype_matched and spec.out_dtypes == outputs_dtype  # whether match output dtypes
                if matched:  # find the lib already impl in_dtypes and out_dtypes
                    break
            lib_not_impl = not matched
        return lib_not_impl

    def _criteria(self):

        _conditions = [
            # insert cast op for lib's dtypes spec
            lambda node, parent_node, edge_tensor:
            (node.type in self.op_need_cast_dtypes_for_lib)
            and (not self.whether_an_inserted_op(parent_node))
            and self._whether_need_adapt_input_dtype(node)
            and (not node.get_param('unquantifiable', optional=True, default_value=False) or
                 (node.get_param('unquantifiable', optional=True, default_value=False) and not is_float(edge_tensor.ir_dtype))),

            # insert cast op for OPs like lstm, gru which ask for specific inputs be quantized with symmetric mode
            lambda node, parent_node, edge_tensor:
            (node.type in ASYM2SYM_OP_DICT)
            and (not self.whether_an_inserted_op(parent_node))
            and (not node.get_param('unquantifiable', optional=True, default_value=False)),
        ]
        return _conditions

    def set_cast_totype(self, node, acc_priority=False):
        # acc_priority or fps_priority

        def mix_quantization_condition(n):
            return n.parents[0].attrs['q_bits_activation'] != n.children[0].attrs['q_bits_activation']

        output_dtype = []
        cast_child_node = node.children[0]

        for output in cast_child_node.outputs:
            if mix_quantization_condition(node) and not output.qinvariant:
                output_dtype.append(bits2dtype(cast_child_node.attrs['q_bits_activation'], is_signed(output.dtype)))
            else:
                output_dtype.append(output.dtype)

        # get the candidates spec which spec_output_dtypes == cast_child_node_output_dtypes
        dtype_spec = cast_child_node.get_lib_dtype_spec()
        candidates = []
        backups = []
        for spec in dtype_spec:
            if len(spec.in_dtypes) != len(cast_child_node.inputs):
                continue
            spec_output_type = spec.out_dtypes
            spec_has_float = any([is_float(dt) for dt in spec.out_dtypes + spec.in_dtypes])
            cur_has_float = any([is_float(t.dtype) for t in cast_child_node.inputs + cast_child_node.outputs])
            if spec_has_float ^ cur_has_float:
                continue
            if output_dtype == spec_output_type:
                candidates.append([spec.in_dtypes, 0.0])
            backups.append([spec.in_dtypes, 0.0])

        if len(candidates) < 1 and len(dtype_spec) > 0:
            candidates = backups
            OPT_WARN(f"layer '{cast_child_node}' asked output dtypes '{output_dtype}' not existed in "
                     f"lib's spec list: {dtype_spec}, you may need to change this layer's quantization bits "
                     f"or add corresponding output dtype impl ({output_dtype}) to lib.")

        for candidate in candidates:
            # (matched, redundant, signed, insufficient)
            score = [0.0, 0.0, 0.0, 0, 0]
            for j, in_dtype in enumerate(candidate[0]):
                if mix_quantization_condition(node) and not cast_child_node.inputs[j].qinvariant:
                    dt = bits2dtype(cast_child_node.attrs['q_bits_activation'],
                                    is_signed(cast_child_node.inputs[j].dtype))
                else:
                    dt = cast_child_node.inputs[j].dtype
                    for cp in cast_child_node.parents:
                        if cp.type == OpType.Cast and cp.additional and cp.outputs[0] == node.children[0].inputs[j]:
                            dt = cp.inputs[0].dtype
                            break
                dqmin, dqmax = dtype2range(dt)
                iqmin, iqmax = dtype2range(in_dtype)
                if iqmin == dqmin and dqmax == iqmax:
                    # matched
                    # may increase the weight of higher bits dtype
                    score[0] += (dqmax - dqmin)
                elif iqmin <= dqmin and dqmax <= iqmax:
                    # redundant
                    score[1] += (dqmin - iqmin) + (iqmax - dqmax)
                else:
                    # insufficient
                    if iqmin > dqmin:
                        score[3] += iqmin - dqmin
                    if dqmax > iqmax:
                        score[3] += dqmax - iqmax
                if is_signed(dt) == is_signed(in_dtype):
                    score[2] += 2
                elif not is_signed(dt) and is_signed(in_dtype):
                    score[2] += 1
                else:
                    pass
            candidate[1] = score
        best = []
        if len(candidates) > 0:
            # more matched, match signed, less redundant, less insufficient,
            try:
                if acc_priority:
                    best = sorted(candidates, key=lambda x: (x[1][0], x[1][2], x[1][1], -x[1][3]))[-1][0]
                else:
                    best = sorted(candidates, key=lambda x: (x[1][0], x[1][2], -x[1][1], -x[1][3]))[-1][0]
            except Exception as e:
                raise e
        return best

    def __call__(self):
        for n in self.g.nodes:
            inserted_ops = []
            need_update_quantization = False
            for inp in n.inputs:
                parent_node = inp.pnode
                for cond in self._criteria():
                    if cond(n, parent_node, inp):
                        inserted_op = self.insert_cast_at_edge(n, parent_node, inp, OpType.Cast)
                        inserted_ops.append(inserted_op)
                        break
            for cast_n in inserted_ops:
                cast_totype_list = self.set_cast_totype(cast_n)
                if len(cast_totype_list):
                    for parent in n.parents:
                        if self.whether_an_inserted_op(parent):
                            inp_id = n.inputs.index(parent.outputs[0])
                            parent.params['to_dtype'] = cast_totype_list[inp_id]
                            parent.outputs[0].dtype = cast_totype_list[inp_id]
                            if parent.params['to_dtype'] != parent.inputs[0].dtype:
                                need_update_quantization = True
                                OPT_WARN(f"'{parent.parents[0]}', cast its output '{parent.inputs[0].name}' "
                                         f"dtype from {parent.inputs[0].dtype} to {parent.params['to_dtype']} for {cast_n.children[0]} "
                                         f"due to lib's {cast_n.children[0].type} spec by insert a cast layer.")
                break

            """
            when trigger_float_op is true, first set_unquantifiable(), then insert the dequantize and quantize op,
            if needed insert cast op, the cast op does not have 'unquantifiable' params, so we independently
            set this params in here
            """
            for cast_n in inserted_ops:
                cast_n.params['unquantifiable'] = any([is_float(dt)
                                                      for dt in [cast_n.inputs[0].dtype, cast_n.params['to_dtype']]])
                if cast_n.params['unquantifiable']:
                    cast_n.outputs[0].ir_dtype = cast_n.params['to_dtype']

            # jira cal-3352
            if need_update_quantization:
                from queue import Queue
                q = Queue(maxsize=0)
                q.put(n)
                while(q.qsize()):
                    dn = q.get()
                    qn = dn.clone()
                    qn.params['unquantifiable'] = False
                    qn.quantize()
                    for k, t in dn.constants.items():
                        if k in qn.constants.keys():
                            tc = qn.constants[k]
                            t.clone_qinfo(tc)
                    for i, t in enumerate(dn.placeholders):
                        tc = qn.placeholders[i]
                        t.clone_qinfo(tc)
                    nc_set = set()
                    for i, t in enumerate(dn.outputs):
                        tc = qn.outputs[i]
                        if tc.dtype != t.dtype:
                            for nchild in dn.children:
                                if t in nchild.inputs:
                                    nc_set.add(nchild)
                        t.clone_qinfo(tc)
                    for nchild in nc_set:
                        q.put(nchild)

        for n in self.g.nodes:
            if n.type in ASYM2SYM_OP_DICT:
                for idx in range(len(n.inputs)):
                    parent = n.parents[idx]
                    if idx in ASYM2SYM_OP_DICT[n.type][0] and QuantMode.is_asymmetric(
                            parent.attrs["q_mode_activation"]):
                        parent.attrs["q_mode_activation"] = ASYM2SYM_OP_DICT[n.type][1]

            if n.type == OpType.MatMul:
                """
                16bits matmul only supports symmetric quantization inputs, and check_quantization_info pass has assured
                16bits activation using symmetric quantization
                """
                for p in n.parents:
                    act_bits = p.attrs['q_bits_activation']
                    act_mode = p.attrs['q_mode_activation']
                    if act_bits >= 16 and QuantMode.is_asymmetric(act_mode):
                        p.attrs['q_mode_activation'] = QuantMode.to_symmetric(act_mode)
