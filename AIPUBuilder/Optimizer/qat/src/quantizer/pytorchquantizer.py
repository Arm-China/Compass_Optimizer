# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import os
import copy
import torch
from torch.fx import Tracer, Graph, Node
from torch.fx.graph_module import GraphModule
import numpy as np

import operator
from operator import eq, add, getitem
from typing import Callable, Union, Tuple, Dict, List, Any
from collections import OrderedDict

from .basequantizer import QATBaseQuantizer
from ..qatlogger import QAT_INFO, QAT_ERROR, QAT_DEBUG, QAT_WARN
from ..qatregister import get_default_fusion_patterns, get_compass_supported_operators, QAT_COMPASS_OPERATORS
from ..qinfo import QuantStage
from ..fuser import is_match
from ..utils import check_result
from ..ops import QInput, QBaseOperator


class AIPUGraphModule(GraphModule):
    def __init__(self, root, graph, class_name="AIPUGraphModule"):
        super().__init__(root, graph, class_name)
        self.class_name = class_name

    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        graph_copy = copy.deepcopy(self.graph)
        return AIPUGraphModule(fake_mod, graph_copy, self.class_name)


class AIPUTracer(Tracer):
    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str):
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True

        if hasattr(m, "_is_leaf_module") and m._is_leaf_module:
            return True

        return m.__module__.startswith("torch.nn") and not isinstance(
            m, torch.nn.Sequential
        )


class PytorchQuantizer(QATBaseQuantizer):

    def __init__(self, config, model=None):
        super().__init__(config)
        self.config = config
        self.model = model
        # self.concrete_args = self.config.get('concrete_args', None)
        self.concrete_args = None
        self.input_shapes = self.config.get('input_shape')
        self.device = self.config.get("device")
        if self.model is None:
            input_model_path = self.config.get('input_model')
            self.model = torch.load(input_model_path, map_location=self.device)
            self.model.eval()
        self.graph_module = self.get_graph_module(self.model, self.concrete_args, inplace=False) if not isinstance(
            self.model, GraphModule) else self.model
        self.modules = dict(self.graph_module.named_modules())
        self.fused_module = None

        self.output_dir = self.config.get("output_dir")
        self.model_name = self.config.get("model_name")

        self.input_dtypes = self.config.get('input_dtype')

    def get_graph_module(self, model, concrete_args, inplace=True):
        if not inplace:
            model = copy.deepcopy(model)

        if isinstance(model, GraphModule):
            traced_model = AIPUGraphModule(model.root, model.graph)
        elif isinstance(model, torch.nn.Module):
            try:
                aipu_tracer = AIPUTracer()
                graph = aipu_tracer.trace(model, concrete_args)
                traced_model = AIPUGraphModule(aipu_tracer.root, graph)
            except Exception as e:
                QAT_ERROR(e)
        else:
            traced_model = None
            raise ValueError("model must be a torch.nn.Module or a torch.fx.GraphModule")

        return traced_model

    def _get_input_dtypes(self, graph_module=None):
        if self.input_dtypes == "":
            from AIPUBuilder.Optimizer.framework import Dtype
            from AIPUBuilder.Optimizer.utils import nptype2dtype, torch_type2dtype
            if self.config.get('train_dataloader', None) is not None or self.config.get('evaluate_dataloader', None) is not None:
                if self.config.get('train_dataloader', None) is not None:
                    # idx 0, [data, label]
                    data_list = self.config.get('train_dataloader')
                else:
                    data_list = self.config.get('evaluate_dataloader')
                for _, data_label in enumerate(data_list):
                    data = data_label[0]
                    self.input_dtypes = [Dtype.FP32] * len(data)
                    for idx, td in enumerate(data):
                        if isinstance(td, np.ndarray):
                            self.input_dtypes[idx] = nptype2dtype(td.dtype)
                        elif isinstance(td, torch.Tensor):
                            self.input_dtypes[idx] = torch_type2dtype(td.dtype)
                        else:
                            QAT_ERROR(
                                f"unsupported object in train/metric data, we now only support np.ndarray or torch.tensor, but now is {type(td)}")
                    break
            else:
                # set default Dtype.FP32
                if graph_module is not None:
                    self.input_dtypes = []
                    for node in graph_module.graph.nodes:
                        if node.op == 'placeholder':
                            self.input_dtypes.append(Dtype.FP32)
                    QAT_WARN(f"Doesnot decide the input dtype, so we use the default float32 dtype")
                else:
                    QAT_ERROR(f"Doesnot decide the all input dtypes")
        else:
            from AIPUBuilder.Optimizer.utils import str2dtype
            for idx, dtype in enumerate(self.input_dtypes):
                if isinstance(dtype, str):
                    self.input_dtypes[idx] = str2dtype(dtype)
        return self.input_dtypes

    def _get_input_shapes(self):
        if self.input_shapes == "":
            if self.config.get('train_dataloader', None) is not None or self.config.get('evaluate_dataloader', None) is not None:
                if self.config.get('train_dataloader', None) is not None:
                    data_list = self.config.get('train_dataloader')
                else:
                    data_list = self.config.get('evaluate_dataloader')
                for _, data_label in enumerate(data_list):
                    data = data_label[0]
                    self.input_shapes = [0] * len(data)
                    for idx, td in enumerate(data):
                        if isinstance(td, (np.ndarray, torch.Tensor)):
                            self.input_shapes[idx] = list(td.shape)
                        else:
                            QAT_ERROR(
                                f"unsupported object in train/metric data, we now only support np.ndarray or torch.tensor, but now is {type(td)}")
                    break
        return self.input_shapes

    def _find_matches(self, root, graph, patterns):
        modules = dict(root.named_modules())
        match_map = {}

        def apply_match(pattern, node, match):
            if isinstance(pattern, tuple):
                s, *args = pattern
                apply_match(s, node, match)
                for subpattern, arg in zip(args, node.args):
                    apply_match(subpattern, arg, match)
            else:
                # the first pattern matches will take precedence
                if node.name not in match_map:
                    match_map[node.name] = match

        for node in reversed(graph.nodes):
            if node.name not in match_map:
                for pattern, value in patterns.items():
                    if is_match(modules, node, pattern):
                        apply_match(pattern, node, (node, value(self, node)))

        return match_map

    def before_fuse_pass(self):
        graph_changed = False
        modules = dict(self.graph_module.named_modules())
        nodes = self.graph_module.graph.nodes
        for node in nodes:
            if node.op == 'placeholder':
                with self.graph_module.graph.inserting_after(node):
                    qinput = QInput()
                    name = node.name + '_QInput'
                    qinput.name = name
                    self.graph_module.add_module(name, qinput)
                    # next_node = node.next
                    new_node = self.graph_module.graph.call_module(name,)
                    node.replace_all_uses_with(new_node)
                    new_node.insert_arg(0, node)
                graph_changed = True
            if node.op == 'call_method' and node.target == 'reshape':
                with self.graph_module.graph.inserting_after(node):
                    args = node.args
                    new_node = self.graph_module.graph.call_function(torch.reshape, args=args)
                    new_node.name = node.name + '_QReshape'
                    node.replace_all_uses_with(new_node)
                    self.graph_module.graph.erase_node(node)
            if node.op == 'call_method' and node.target == 'permute':
                with self.graph_module.graph.inserting_after(node):
                    args = node.args
                    new_node = self.graph_module.graph.call_function(torch.permute, args=args)
                    new_node.name = node.name + '_QTranspose'
                    node.replace_all_uses_with(new_node)
                    self.graph_module.graph.erase_node(node)
            if node.op == 'call_method' and node.target == 'expand':
                from ..ops import QExpand
                with self.graph_module.graph.inserting_after(node):
                    size = []
                    local_args = (node.args[0], )
                    if len(node.args) > 1:
                        for arg in node.args[1:]:
                            if not isinstance(arg, int):
                                local_args = node.args
                                break
                    size = [] if len(local_args) > 1 else node.args[1:]
                    qexpand = QExpand(size=size)
                    name = node.name + '_QExpand'
                    self.graph_module.add_module(name, qexpand)
                    new_node = self.graph_module.graph.call_module(name, args=local_args)
                    node.replace_all_uses_with(new_node)
                    self.graph_module.graph.erase_node(node)
                graph_changed = True
            if node.op == 'get_attr':
                '''
                when serialize, do this transform from get_attr to QConstant, because if now do QConstant,
                the QConstant forward does not call by anyother module.
                '''
                pass
                # from ..ops import QConstant
                # name = node.name + '_QConstant'
                # target = node.target
                # cur_module = self.graph_module
                # if '.' in target:
                #     cur_module_target, target = target.strip().rsplit('.', 1)
                #     cur_module = self.graph_module.get_submodule(cur_module_target)
                # data = cur_module.__getattr__(target)
                # qconstant = QConstant(name=name, data=data)
                # self.graph_module.add_module(name, qconstant)
                # new_node = self.graph_module.graph.call_module(name, args=node.args)
                # node.replace_all_uses_with(new_node)
                # self.graph_module.graph.erase_node(node)
                # graph_changed = True

        if graph_changed:
            self.graph_module.recompile()
            self.graph_module = AIPUGraphModule(self.graph_module, self.graph_module.graph)

    def after_fuse_pass(self):
        self.modules = dict(self.fused_module.named_modules())
        for fx_node in self.fused_module.graph.nodes:
            # set prev_node for getting the input_scale to finetune biases data or other usage
            if fx_node.target in self.modules and isinstance(self.modules[fx_node.target], QBaseOperator):
                prev_nodes = fx_node.all_input_nodes
                prev_ms = []
                for pn in prev_nodes:
                    if pn.target in self.modules:
                        prev_ms.append(self.modules[pn.target])
                if len(prev_ms) and len(prev_ms) == len(prev_nodes):
                    self.modules[fx_node.target].set_prev_modules(prev_ms)
            # set unquantifiable attributes

    def _fuse(self, graph_module=None):
        if graph_module is None:
            graph_module = self.graph_module
        self.modules = dict(graph_module.named_modules())

        graph_changed = False
        modules = dict(graph_module.named_modules())
        fusion_patterns = get_default_fusion_patterns()
        fusion_pairs = self._find_matches(
            graph_module, graph_module.graph, fusion_patterns
        )

        fused_graph = graph_module.graph
        env = {}

        for node in graph_module.graph.nodes:
            root_node, obj = fusion_pairs.get(node.name, (None, None))
            if root_node is node:
                assert obj is not None
                env[node.name] = obj.fuse(graph_module, modules)
                graph_changed = True

        if graph_changed:
            graph_module = AIPUGraphModule(graph_module, fused_graph)

            float_output = self.get_output(self.model)
            fused_output = self.get_output(graph_module)
            ret = check_result(float_output, fused_output)
            if not ret:
                QAT_ERROR(f"check origin model and fused model failed")
        return graph_module, graph_changed

    def fuse(self):
        # QAT_DEBUG(f"before fuse orig module:\n{self.graph_module.graph.print_tabular()}")
        self.before_fuse_pass()
        self.fused_module, is_fused = self._fuse(self.graph_module)
        while is_fused:
            self.fused_module, is_fused = self._fuse(self.fused_module)
        self.after_fuse_pass()
        # print(self.fused_module.graph.print_tabular())
        # print(self.fused_module.code)

        return self.fused_module

    def gen_input_data(self, input_shapes, input_dtypes):
        from AIPUBuilder.Optimizer.utils import dtype2torch_type
        input_tensor = []
        for idx, shape in enumerate(input_shapes):
            dtype = input_dtypes[idx]
            input_tensor.append(torch.zeros(*shape, device=self.device).to(dtype2torch_type(dtype)))
        return input_tensor

    def get_output(self, model=None, input_data=None):
        self._get_input_shapes()
        self._get_input_dtypes(model)
        if input_data is None:
            input_data = self.gen_input_data(self.input_shapes, self.input_dtypes)

        if model is None:
            model = self.model

        model.to(self.device)
        out = self.forward(model, input_data)
        return out

    def all_node_args_have_no_tensors(self, node, modules, cache):
        """
        If we know for sure that all of this node's args have no
        tensors (are primitives), return True.  If we either
        find a tensor or are not sure, return False. Note: this
        function is not exact.
        """
        if cache and node in cache:
            return cache[node]

        result = False  # will be overwritten
        if not isinstance(node, Node):
            result = True
        elif node.op == "placeholder":
            result = False
        # elif node.op == "call_module":
        #     assert isinstance(node.target, str)
        #     if _is_activation_post_process(modules[node.target]):
        #         result = all_node_args_have_no_tensors(node.args[0], modules, cache)  # type: ignore[arg-type]
        # elif node.op == "call_module":
        #     result = False
        elif node.op == "call_function" and node.target is operator.getitem:
            result = self.all_node_args_have_no_tensors(node.args[0], modules, cache)  # type: ignore[arg-type]
        elif node.op == "get_attr":
            result = False
        elif node.target is getattr and node.args[1] in ["ndim", "shape"]:
            # x1 = x0.ndim
            result = True
        elif node.op == "call_method" and node.target == "size":
            # x1 = x0.size(0)
            result = True
        else:
            found_one_tensor = False
            for arg in node.args:
                if isinstance(arg, list):
                    for list_el in arg:
                        if isinstance(list_el, Node):
                            this_list_el_args_have_no_tensors = (
                                self.all_node_args_have_no_tensors(list_el, modules, cache)
                            )
                            found_one_tensor = found_one_tensor or (
                                not this_list_el_args_have_no_tensors
                            )
                            # If found_one_tensor is True, there is no point in
                            # recursing further as the end result will always
                            # be True.
                            # TODO(future PR): remove this entire function  and
                            # change to dtype inference without recursion.
                            if found_one_tensor:
                                result = not found_one_tensor
                                if cache:
                                    cache[node] = result
                                return result
                elif isinstance(arg, int):
                    pass
                else:
                    if isinstance(arg, Node):
                        this_arg_args_have_no_tensors = self.all_node_args_have_no_tensors(
                            arg, modules, cache
                        )
                        found_one_tensor = found_one_tensor or (
                            not this_arg_args_have_no_tensors
                        )
                        # If found_one_tensor is True, there is no point in
                        # recursing further as the end result will always
                        # be True.
                        # TODO(future PR): remove this entire function  and
                        # change to dtype inference without recursion.
                        if found_one_tensor:
                            result = not found_one_tensor
                            if cache:
                                cache[node] = result
                            return result
                    else:
                        found_one_tensor = True
                result = not found_one_tensor
        if cache:
            cache[node] = result
        return result

    def _build_g(self, model, prefix, input_shapes, ir_mode='fp'):

        _ = self.get_output(model)

        all_modules = dict(model.named_modules())
        torch_name_to_aipu_tensor = {}
        from AIPUBuilder.core import Graph, TensorList
        from ..ops import QInput
        g_output_tensors = []
        module_output_node_num = 0
        input_idx = 0
        with Graph(f"tmp") as aipu_g:
            try:
                cache = {}
                for node in model.graph.nodes:
                    if self.all_node_args_have_no_tensors(node, all_modules, cache):
                        continue
                    if node.op == 'call_function' and eq(node.target, getitem):
                        in_node, idx = node.args
                        if in_node.name not in torch_name_to_aipu_tensor:
                            continue
                        in_node_ops_tensors = torch_name_to_aipu_tensor[in_node.name]
                        if isinstance(in_node_ops_tensors, (list, tuple)):
                            torch_name_to_aipu_tensor.update({node.name: in_node_ops_tensors[idx]})
                        elif isinstance(idx, (list, tuple)) and isinstance(idx[0], slice):
                            from AIPUBuilder import ops
                            ot = ops.crop(in_node_ops_tensors, [[0, 1], [0, 1], [0, 768]])
                            torch_name_to_aipu_tensor.update({node.name: ot})
                        continue

                    if node.op == 'get_attr':
                        from ..ops import QConstant
                        name = node.name + '_QConstant'
                        target = node.target
                        cur_module = self.graph_module
                        if '.' in target:
                            cur_module_target, target = target.strip().rsplit('.', 1)
                            cur_module = self.graph_module.get_submodule(cur_module_target)
                        data = cur_module.__getattr__(target)
                        qconstant = QConstant(name=name, data=data)
                        qconstant.ir_mode = ir_mode
                        out = qconstant.serialize()
                        torch_name_to_aipu_tensor.update({node.name: out})
                        continue

                    if node.op == 'output':
                        prev_node = node.args[0]
                        module_output_node_num += 1
                        if prev_node.name not in torch_name_to_aipu_tensor:
                            QAT_ERROR(f"one output tensor in torch.module is not traversed, please check.")
                        else:
                            g_output_tensors.append(torch_name_to_aipu_tensor[prev_node.name])

                    if node.op != 'call_module':
                        QAT_DEBUG(f"please check {node} is {node.op}, which is not call_module")
                        continue
                    m = all_modules[node.target]
                    if isinstance(m, QInput):
                        m.ir_mode = ir_mode
                        ot = m.serialize(self.input_shapes[input_idx], self.input_dtypes[input_idx])
                        torch_name_to_aipu_tensor.update({node.name: ot})
                        input_idx += 1
                        continue
                    inputs = []
                    for arg in node.args:
                        if isinstance(arg, (tuple, list)):
                            cargs = []
                            for carg in arg:
                                if isinstance(carg, torch.fx.Node):
                                    if not carg.name in torch_name_to_aipu_tensor and carg.op not in ['placeholder']:
                                        QAT_ERROR(f"the input: {arg.name} is no-traversal in node:{node.name}")
                                        break
                                    cargs.append(torch_name_to_aipu_tensor[carg.name])
                            inputs.append(cargs)
                            continue
                        if isinstance(arg, torch.fx.Node):
                            if not arg.name in torch_name_to_aipu_tensor and node.prev.op not in ['placeholder', 'call_function']:
                                QAT_WARN(f"the input: {arg.name} is no-traversal in node:{node.name}")
                                break
                            inputs.append(torch_name_to_aipu_tensor[arg.name])

                    if isinstance(m, torch.nn.Dropout):
                        from AIPUBuilder import ops
                        ot = ops.reshape(*inputs, list(inputs[0].shape))
                        torch_name_to_aipu_tensor.update({node.name: ot})
                        continue

                    try:
                        ot = m.serialize(*inputs)
                    except Exception as e:
                        QAT_ERROR(
                            f"failed node: op = {node.op}, target ={node.target}, name={node.name}, and its module = {m}")
                        QAT_DEBUG(f"now add node.anme and opsapi tensor:")
                        for k, v in torch_name_to_aipu_tensor.items():
                            QAT_DEBUG(k, v)
                        raise e

                    if node.name not in torch_name_to_aipu_tensor:
                        torch_name_to_aipu_tensor.update({node.name: ot})
            except Exception as e:
                aipu_g.serialize_scale_zp = True
                irt = os.path.join(self.output_dir, f"{self.model_name}_{prefix}_{ir_mode}_failed.txt")
                irb = os.path.join(self.output_dir, f"{self.model_name}_{prefix}_{ir_mode}_failed.bin")
                aipu_g.serialize(irt, irb)
                QAT_INFO(
                    f"when serializing the {ir_mode} IR meets the error, which is {e}, you can see the failed IR at: {irt}")
                raise e

        if len(g_output_tensors) == module_output_node_num:
            aipu_g.output_tensors = TensorList(g_output_tensors)

        if ir_mode == 'qat':
            from AIPUBuilder.core import Quantizer
            self._pre_quantize_pass(aipu_g)
            qer = Quantizer(aipu_g)
            qer.quantize()
        self._gsim_graph(aipu_g)
        aipu_g.serialize_scale_zp = True if ir_mode != 'fp' else False
        irt = os.path.join(self.output_dir, f"{self.model_name}_{prefix}_opsapi_{ir_mode}.txt")
        irb = os.path.join(self.output_dir, f"{self.model_name}_{prefix}_opsapi_{ir_mode}.bin")
        aipu_g.serialize(irt, irb)
        QAT_INFO(f"serialize the {ir_mode} IR Done: {irt}")

    def _pre_quantize_pass(self, g):
        from AIPUBuilder.core import OpType
        unify_shift_mode = self.config.get('compat_quantized_model_unify_shifts_mode')

        for node in g.nodes:
            node.attrs['unify_shift_mode'] = unify_shift_mode
            if node.type == OpType.Softmax:
                pass

    def serialize(self, model=None, prefix="", input_shapes=[]):
        if model is None:
            model = self.fused_module
            model.eval()
        ir_mode = 'fp'
        self._build_g(model, prefix, input_shapes, ir_mode)
        ir_mode = 'qat'
        self._build_g(model, prefix, input_shapes, ir_mode)

    def _gsim_graph(self, g):
        from AIPUBuilder.simplifier import fuse_transpose, fuse_reshape_transpose_conv, eliminate_transpose
        _gsim_pass = [
            fuse_transpose,
            fuse_reshape_transpose_conv,
            eliminate_transpose,
        ]
        for gp in _gsim_pass:
            gp(g)

    # def _insert_dequantize_node(self, node: Node, graph: Graph) -> None:
    #     """Inserts dequantize node for `node` in `graph`"""
    #     with graph.inserting_after(node):
    #         dequantize_node = graph.call_method("dequantize", (node,))
    #         for user_node in dict(node.users):
    #             if user_node is not dequantize_node:
    #                 user_node.replace_input_with(node, dequantize_node)

    # Returns a function that can get a new attribute name for module with given
    # prefix, for example,
    # >> get_new_observer_name = get_new_attr_name_with_prefix('_observer')
    # >> new_name = get_new_observer_name(module)
    # new_name will be an unused attribute name on module, e.g. `_observer_1`
    def get_new_attr_name_with_prefix(self, prefix: str) -> Callable:
        prefix = prefix.replace(".", "_")

        def get_new_attr_name(module: torch.nn.Module):
            def get_attr_name(i: int):
                return prefix + str(i)

            i = 0
            attr_name = get_attr_name(i)
            while hasattr(module, attr_name):
                i += 1
                attr_name = get_attr_name(i)
            return attr_name

        return get_new_attr_name

    def create_getattr_from_value(
        self, module: torch.nn.Module, graph: Graph, prefix: str, value: Any
    ) -> Node:
        """
        Given a value of any type, creates a getattr node corresponding to the value and
        registers the value as a buffer to the module.
        """
        get_new_attr_name = self.get_new_attr_name_with_prefix(prefix)
        attr_name = get_new_attr_name(module)
        # device = assert_and_get_unique_device(module)
        new_value = (
            value.clone().detach()
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, device=self.device)
        )
        module.register_buffer(attr_name, new_value)
        # Create get_attr with value
        attr_node = graph.create_node("get_attr", attr_name)
        return attr_node

    def export(self, model=None, output_model_name=''):
        if model is None:
            model = self.fused_module
            model.eval()
        _ = self.get_output(model)

        qoperators = get_compass_supported_operators()
        all_modules = dict(model.named_modules())
        for node in model.graph.nodes:
            if node.op == 'call_module':
                m = all_modules[node.target]
                if type(m) in qoperators and m._use_input_QConfig:
                    input_node = node.args[0]
                    if isinstance(input_node, Node) and input_node.op == 'call_module':
                        input_in_m = all_modules[input_node.target]
                        if type(input_in_m) in qoperators:
                            m.activation_qinfo = copy.deepcopy(input_in_m.activation_qinfo)

        all_modules = dict(model.named_modules())
        graph = model.graph
        idx = 0
        for node in graph.nodes:
            new_args = []
            if node.op != 'call_module':
                continue
            # cur_node_m = all_modules[in_node.target]
            for arg_idx, in_node in enumerate(node.args):
                new_args.append(in_node)
                if isinstance(in_node, Node) and in_node.op == "call_module":
                    m = all_modules[in_node.target]
                    if type(m) in qoperators:
                        # aipu qoperator
                        if hasattr(m, 'activation_qinfo'):
                            '''
                            TODO
                            when split/slice module, will generate multi-outputs, should split the act_qinfos,
                            but this does not work, because split outputs a tuple of tensor,
                            its child must have getitem to get one used tensor. so if want to support this, getitem should be fused
                            to a Module(inherited from QBaseOperator). and serialize method ....
                            '''
                            act_qinfos = m.activation_qinfo if isinstance(
                                m.activation_qinfo, (tuple, list)) else [m.activation_qinfo]
                        else:
                            QAT_WARN(
                                f"op={node.op}, target={node.target}, qoperator={type(m)} does not have 'activation_qinfo', and will keep float type.")
                        for _, act_q in enumerate(act_qinfos):
                            if act_q.qinvariant:
                                continue
                            qparams = act_q.get_qparams()
                            if len(qparams) == 0:
                                QAT_ERROR(
                                    f"when getting qinfo params failed at in_node:{in_node}, in_node's module:{m}, and cur node:{node}")
                                continue
                            quantize_op = None
                            node_type = "call_function"
                            if act_q.scale.numel() == 1:
                                quantize_op = torch.quantize_per_tensor
                            else:
                                quantize_op = torch.quantize_per_channel
                            with graph.inserting_before(node):
                                quantize_op_inputs = [in_node]
                                for key, value_or_node in qparams.items():
                                    if key in ["_scale_", "_zero_point_"]:
                                        qparam_node = self.create_getattr_from_value(
                                            model, graph, m.name + key + f"{idx}", value_or_node)
                                        idx += 1
                                        quantize_op_inputs.append(qparam_node)
                                    else:
                                        # for qparams that are not scale/zero_point (like axis, dtype) we store them as literals in the graph.
                                        quantize_op_inputs.append(value_or_node)
                                quantized_node = graph.create_node(node_type, quantize_op, (), {})
                                dequantized_node = graph.call_method("dequantize", ())
                                quantized_node.args = tuple(quantize_op_inputs)
                                dequantized_node.args = (quantized_node, )
                                new_args[arg_idx] = dequantized_node
            node.args = tuple(new_args)

            if node.op == 'call_module':
                m = all_modules[node.target]
                new_args = list(node.args)
                if type(m) in qoperators:
                    m_key = 'weight'
                    if hasattr(m, m_key):
                        if not hasattr(m, f"{m_key}_qinfo"):
                            QAT_ERROR(f"{m_key}_qinfo is not in {m}, which fails to generate deq_{key} mode")
                        const_v = m.weight
                        qinfo = m.weight_qinfo
                        qparams = m.weight_qinfo.get_qparams()
                        if len(qparams) == 0:
                            QAT_ERROR(f"{m}, {node}")
                            continue
                        scale = qinfo.scale
                        node_type = 'call_function'
                        quantize_op = None
                        if scale.numel() == 1:
                            quantize_op = torch.quantize_per_tensor
                        else:
                            quantize_op = torch.quantize_per_channel
                        qparams.update({'weight': const_v})
                        qparams.move_to_end('weight', last=False)
                        quantize_op_inputs = []
                        with graph.inserting_before(node):
                            for key, value_or_node in qparams.items():
                                if key in ['weight', '_scale_', '_zero_point_']:
                                    qparam_node = self.create_getattr_from_value(
                                        model, graph, m.name + key + f"{idx}", value_or_node
                                    )
                                    idx += 1
                                    quantize_op_inputs.append(qparam_node)
                                else:
                                    # for qparams that are not scale/zero_point (like axis, dtype) we store them as literals in the graph.
                                    quantize_op_inputs.append(value_or_node)
                            quantized_node = graph.create_node(node_type, quantize_op, (), {})
                            dequantized_node = graph.call_method("dequantize", ())
                            quantized_node.args = tuple(quantize_op_inputs)
                            dequantized_node.args = (quantized_node, )
                            m.weight = None
                            new_args.append(dequantized_node)

                    m_key = 'bias'
                    if hasattr(m, m_key) and getattr(m, m_key) is not None:
                        # now save float bias
                        with graph.inserting_before(node):
                            qparam_node = self.create_getattr_from_value(
                                model, graph, m.name + f"bias{idx}", m.bias
                            )
                            m.bias = None
                            new_args.append(qparam_node)
                    node.args = tuple(new_args)

        model.graph.eliminate_dead_code()
        exported_module = GraphModule(model, model.graph)

        for m in model.modules():
            if isinstance(m, tuple(QAT_COMPASS_OPERATORS.keys())):
                m.quant_stage = QuantStage.str_to_quantstage('fp32')

        if output_model_name == '':
            output_model_name = os.path.join(self.output_dir, f"{self.model_name}_onnx.onnx")
        dummy_inputs = self.gen_input_data(self.input_shapes, self.input_dtypes)

        self.input_names = self.config.get('input_name', None)
        self.output_names = self.config.get('output_name', None)
        torch.onnx.export(exported_module,
                          args=tuple(dummy_inputs),
                          f=output_model_name,
                          input_names=self.input_names,
                          output_names=self.output_names)
