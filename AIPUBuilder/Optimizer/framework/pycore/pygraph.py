# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3

import torch
__all__ = [
    "PyGraph",
]


class PyGraphView:
    def __init__(self):
        self.nodes = []
        self.inflow_tensors = ()
        self.outflow_tensors = ()


class PyGraph:
    def __init__(self, name="unamed"):
        self.name = str(name)
        self.net_ = None  # take advantage of networkx's basic graph algorithms
        self.nodes = []
        self.input_tensors = ()
        self.output_tensors = ()
        self.ref_count_tensors = {}

    def subgraph_view(self, nodes):
        from AIPUBuilder.Optimizer.framework.pycore.pytype import OpType
        gview = PyGraphView()
        ilist = []
        olist = []
        for n in self.nodes:
            if n in nodes:
                gview.nodes.append(n)
                for it in n.inputs:
                    if it.pnode not in nodes:
                        ilist.append(it)
            else:
                for it in n.inputs:
                    if it.pnode in nodes:
                        olist.append(it)
        gview.inflow_tensors = tuple(ilist)
        gview.outflow_tensors = tuple(olist)
        return gview

    def clone(self):
        import copy
        from AIPUBuilder.Optimizer.framework.pycore.pynode import PyNode
        g = self.__class__(self.name)
        nmap = {}
        emap = {}
        for n in self.nodes:
            pn = PyNode(n.name, n.type)
            for k, v in n.params.items():
                pn.params[k] = copy.deepcopy(v)
            for k, v in n.attrs.items():
                pn.attrs[k] = copy.deepcopy(v)
            for k, v in n.constants.items():
                pv = v.clone(v.name)
                pn.constants[k] = pv
            for v in n.placeholders:
                pv = v.clone(v.name)
                pn.placeholders.append(pv)
            g.nodes.append(pn)
            pn.graph = g
            nmap[pn.name] = pn
            # store edges
            for v in n.outputs:
                pv = v.clone(v.name)
                emap[pv.name] = pv
        # connect edges
        for i, n in enumerate(self.nodes):
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
        ilist = []
        for t in self.input_tensors:
            ilist.append(emap[t.name])
        g.input_tensors = tuple(ilist)
        olist = []
        for t in self.output_tensors:
            olist.append(emap[t.name])
        g.output_tensors = tuple(olist)
        g.init_networkx()

        return g

    def tensors(self, tname=None):
        from AIPUBuilder.Optimizer.logger import OPT_DEBUG, OPT_WARN
        tlist = []
        tn_pairs = []
        # priority 1
        for n in self.nodes:
            for t in n.outputs:
                tlist.append(t)
                tn_pairs.append((t, n))
        # priority 2
        for n in self.nodes:
            for t in n.constants.values():
                tlist.append(t)
                tn_pairs.append((t, n))
        # priority 3
        for n in self.nodes:
            for t in n.placeholders:
                tlist.append(t)
                tn_pairs.append((t, n))
        if tname is not None:
            hits = []
            for t, n in tn_pairs:
                if tname == t.name:
                    hits.append((t, n))
            if len(hits) < 1:
                OPT_DEBUG('can not find tensor "%s" in graph: %s ' % (tname, self.name))
                return None
            else:
                if len(hits) > 1:
                    msg = 'find %d tensors named "%s", and these namesakes exist in :' % (len(hits), tname)
                    for t, n in hits:
                        msg += '\n layer_id=%s, layer_type=%s, layer_name=%s' % (
                            str(n.attrs['layer_id']), str(n.type), str(n.name))
                    OPT_WARN(msg)
                return hits[0][0]
        else:
            return tlist

    def get_valid_node_name(self, name: str) -> str:
        names_pool = set([n.name for n in self.nodes])
        new_name = self.get_valid_name(name, names_pool)
        return new_name

    def get_valid_tensor_name(self, name: str) -> str:
        names_pool = set([t.name for t in self.tensors()])
        new_name = self.get_valid_name(name, names_pool)
        return new_name

    def get_valid_name(self, name, names_pool):
        if name not in names_pool:
            return name

        base_name = name
        new_name = name
        idx = 0
        if name in names_pool:
            pos = name.rfind('_')
            if pos != -1:
                sub_name = name[pos+1:]
                if sub_name.isdigit():
                    base_name = name[:pos]
                    idx = int(sub_name) + 1
            new_name = f"{base_name}_{idx}"
            while new_name in names_pool:
                idx += 1
                new_name = f"{base_name}_{idx}"
        return new_name

    def init_networkx(self):
        import networkx as nx

        def topological_generations(G):
            if not G.is_directed():
                raise nx.NetworkXError("Topological sort not defined on undirected graphs.")

            multigraph = G.is_multigraph()
            indegree_map = {v: d for v, d in G.in_degree() if d > 0}
            zero_indegree = [v for v, d in G.in_degree() if d == 0]

            while zero_indegree:
                this_generation = zero_indegree
                zero_indegree = []
                for node in this_generation:
                    if node not in G:
                        raise RuntimeError("Graph changed during iteration")
                    for child in G.neighbors(node):
                        try:
                            indegree_map[child] -= len(G[node][child]) if multigraph else 1
                        except KeyError as e:
                            raise RuntimeError("Graph changed during iteration") from e
                        if indegree_map[child] == 0:
                            zero_indegree.append(child)
                            del indegree_map[child]
                yield this_generation

            if indegree_map:
                raise nx.NetworkXUnfeasible(
                    "Graph contains a cycle or graph changed during iteration"
                )

        net = nx.DiGraph()
        net.add_nodes_from(self.nodes)
        # restore edges by unique tensor name
        edges = {}
        for n in self.nodes:
            for t in n.outputs:
                edges[t.name] = [n, ]
        for n in self.nodes:
            for t in n.inputs:
                if t.name in edges.keys():
                    edges[t.name].append(n)
        for k, v in edges.items():
            for i in range(1, len(v)):
                net.add_edge(v[0], v[i])
        # keep the orders of node's parents and children consistent with its inputs and outputs
        for n in self.nodes:
            parents = []
            # a layer's input can be only one another layer's output
            for t in n.inputs:
                for p in net.predecessors(n):
                    flag = False
                    for x in p.outputs:
                        if t.name == x.name:
                            parents.append(p)
                            flag = True
                            break
                    if flag:
                        break
            n.parents = tuple(parents)
            children = []
            # a layer's output can be many other layers' input
            for t in n.outputs:
                for d in net.successors(n):
                    for x in d.inputs:
                        if t.name == x.name:
                            children.append(d)
                            break
            n.children = tuple(children)

        pre_idx_map = {}
        for k, n in enumerate(self.nodes):
            pre_idx_map[n] = k
        topological_nodes = []
        for i, g in enumerate(topological_generations(net)):
            for n in sorted(g, key=lambda xn: pre_idx_map[xn]):
                n.attrs['tgid'] = i
                n.graph = self
                for ot in n.outputs:
                    ot.pnode = n
                topological_nodes.append(n)
        self.nodes = topological_nodes

        self.net_ = net
        return self.net_

    def reset_edge_tensors_ref_count(self):
        ref_count_tensors = {}
        for n in self.nodes:
            for it in n.inputs:
                if it.name not in ref_count_tensors.keys():
                    ref_count_tensors[it.name] = [1, it]
                else:
                    ref_count_tensors[it.name][0] += 1
        self.ref_count_tensors = ref_count_tensors

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)
            self.init_networkx()

    def remove_node(self, node):
        if node in self.nodes:
            node.graph = None
            node.parents = ()
            node.children = ()
            inputs = []
            for i, t in enumerate(node.inputs):
                inputs.append(t.clone())
            node.inputs = tuple(inputs)
            outputs = []
            for i, t in enumerate(node.outputs):
                outputs.append(t.clone())
            node.outputs = tuple(outputs)
            idx = self.nodes.index(node)
            self.nodes = self.nodes[:idx] + self.nodes[idx+1:]
            self.init_networkx()

    def replace_node_safely(self, old, new):
        if old in self.nodes:
            idx = self.nodes.index(old)
            self.nodes[idx] = new
            new.inputs = old.inputs
            new.outputs = old.outputs
            new.parents = old.parents
            new.children = old.children
            new.graph = self
            if 'tgid' in old.attrs:
                new.attrs['tgid'] = old.attrs['tgid']
                old.attrs.pop('tgid')
            old.parents = ()
            old.children = ()
            old.inputs = ()
            old.outputs = ()
            old.graph = None
            for i, t in enumerate(new.inputs):
                old.inputs = old.inputs + (t.clone(),)
            for i, t in enumerate(new.outputs):
                old.outputs = old.outputs + (t.clone(),)

    def cut_subgraph(self, nodes):
        sg = self.subgraph_view(nodes)
        clone_t_inp = {}
        clone_t_out = {}
        for n in sg.nodes:
            n.graph = None
            for idx, it in enumerate(n.inputs):
                if it in sg.inflow_tensors:
                    it.pnode = None
                    info = (n, idx)
                    if it in clone_t_inp.keys():
                        clone_t_inp[it].append(info)
                    else:
                        clone_t_inp[it] = [info, ]
            nplist = []
            for np in n.parents:
                if np in sg.nodes:
                    nplist.append(np)
            n.parents = tuple(nplist)
            for idx, ot in enumerate(n.outputs):
                if ot in sg.outflow_tensors:
                    ot.pnode = None
                    info = (n, idx)
                    if ot in clone_t_out.keys():
                        clone_t_out[ot].append(info)
                    else:
                        clone_t_out[ot] = [info, ]
            nclist = []
            for nc in n.children:
                if nc in sg.nodes:
                    nclist.append(nc)
            n.children = tuple(nclist)
        ilist = []
        olist = []
        for t, nlist in clone_t_inp.items():
            # with the same name for matching in the future
            tt = t.clone(t.name)
            ilist.append(tt)
            for n, idx in nlist:
                n.replace_input_temporarily(idx, tt)
        for t, nlist in clone_t_out.items():
            # with the same name for matching in the future
            tt = t.clone(t.name)
            olist.append(tt)
            for n, idx in nlist:
                n.replace_output_temporarily(idx, tt)
        sg.inflow_tensors = tuple(ilist)
        sg.outflow_tensors = tuple(olist)

        left_nodes = []
        for n in self.nodes:
            if n not in sg.nodes:
                left_nodes.append(n)
        self.nodes = left_nodes
        return sg

    def forward(self, feed_data, disable_pbar=True, keep_tensors=False):
        return self.forward_to(None, feed_data, disable_pbar, keep_tensors)

    def forward_to(self, dest_node, feed_data, disable_pbar=True, keep_tensors=False):
        from AIPUBuilder.Optimizer.framework.pycore.pytensor import PyTensor
        if keep_tensors:
            self.ref_count_tensors = {}
        else:
            self.reset_edge_tensors_ref_count()
        data = feed_data
        if len(self.input_tensors) == 1 and not isinstance(feed_data, list):
            data = [feed_data, ]
        for inp, d in zip(self.input_tensors, data):
            inp.betensor = PyTensor('tmp', d).betensor.to(inp.assigned_device)

        import sys
        from AIPUBuilder.Optimizer.logger import tqdm
        with tqdm(total=len(self.nodes), desc='forward_to', file=sys.stdout, leave=True, disable=disable_pbar) as pbar:
            for n in self.nodes:
                if len(n.inputs) > 0:
                    device = n.outputs[0].assigned_device
                    for inp in n.inputs:
                        if not inp.assigned_device == device:
                            inp.betensor = inp.betensor.to(device)
                            for attrname in dir(inp):
                                attr = getattr(inp, attrname)
                                if isinstance(attr, torch.Tensor):
                                    setattr(inp, attrname, attr.to(device))
                n.forward()
                if dest_node is not None and n == dest_node:
                    break
                pbar.update(1)
            pbar.refresh()

        ret = []
        if dest_node is None:
            for out in self.output_tensors:
                # ret.append(out.betensor)
                ret.append(out)  # better to carry quantization info like scale
        else:
            for out in dest_node.outputs:
                ret.append(out)

        if keep_tensors:
            self.ref_count_tensors = {}
        else:
            tz = PyTensor('null').betensor
            for n in self.nodes:
                for pld in n.placeholders:
                    del pld.betensor
                    pld.betensor = tz
                for t in n.outputs:
                    if t not in self.output_tensors:
                        del t.betensor
                        t.betensor = tz
            self.reset_edge_tensors_ref_count()

        return ret

    def forward_from_src_to_dst(self, src_node, dst_node, disable_pbar=True, keep_tensors=False):
        from AIPUBuilder.Optimizer.framework.pycore.pytensor import PyTensor
        if keep_tensors:
            self.ref_count_tensors = {}
        else:
            self.reset_edge_tensors_ref_count()
        num = len(self.nodes)
        start = 0
        for n in self.nodes:
            start += 1
            if n != src_node:
                continue
            else:
                for k, out in enumerate(n.outputs):
                    out.betensor = src_node.outputs[k].betensor
                break

        import sys
        from AIPUBuilder.Optimizer.logger import tqdm
        with tqdm(total=num-start, desc='forward_from_src_to_dst', file=sys.stdout, leave=True, disable=disable_pbar) as pbar:
            for k in range(start, num):
                n = self.nodes[k]
                n.forward()
                if dst_node is not None and n == dst_node:
                    for k, out in enumerate(n.outputs):
                        dst_node.outputs[k].betensor = out.betensor
                    break
                pbar.update(1)
            pbar.refresh()
        if keep_tensors:
            self.ref_count_tensors = {}
        else:
            tz = PyTensor('null').betensor
            for n in self.nodes:
                for pld in n.placeholders:
                    del pld.betensor
                    pld.betensor = tz
                for t in n.outputs:
                    if t not in self.output_tensors and t not in dst_node.outputs:
                        del t.betensor
                        t.betensor = tz
            self.reset_edge_tensors_ref_count()

    def to_torch_module(self):
        torch_module = GraphModule(self.input_tensors, self.output_tensors)
        for n in self.nodes:
            torch_module.add_module(n.name.replace('.', '_').replace('\\', '_'), LayerModule(n))
        return torch_module

    def export_as_onnx(self, file_path, save_optimized_model=False, use_ir_shape=True):
        from AIPUBuilder.Optimizer.utils import dtype2torch_type
        from AIPUBuilder.Optimizer.framework import PyTensor
        m = self.to_torch_module()
        x = []
        x_names = []
        for inp in self.input_tensors:
            tshape = inp.ir_shape if use_ir_shape else inp.shape
            t = PyTensor(inp.name, torch.zeros(tshape).to(dtype2torch_type(inp.ir_dtype)))
            x.append(t.betensor)
            x_names.append(inp.name)
        y_names = []
        for out in self.output_tensors:
            y_names.append(out.name)
        torch.onnx.export(
            m,
            x,
            file_path,
            input_names=x_names,
            output_names=y_names,
            export_params=True,
            do_constant_folding=False,
            opset_version=15,
        )
        if save_optimized_model:
            import onnxruntime
            sess_options = onnxruntime.SessionOptions()
            sess_options.optimized_model_filepath = file_path + '.optimized.onnx'
            ort_session = onnxruntime.InferenceSession(file_path, sess_options)

    def serialize(self, ir_txt, ir_bin):
        from AIPUBuilder.Optimizer.framework.pycore.pyir import serialize_graph_to_ir
        serialize_graph_to_ir(self, ir_txt, ir_bin)

    def enable_fit_dtype(self, flag: bool):
        for node in self.nodes:
            node.enable_fit_dtype = flag

    @classmethod
    def parse(cls, ir_txt, ir_bin):
        from AIPUBuilder.Optimizer.framework.pycore.pyir import parse_graph_from_ir
        g = parse_graph_from_ir(ir_txt, ir_bin)
        return g

    @classmethod
    def parse_without_weight(cls, ir_txt):
        from AIPUBuilder.Optimizer.framework.pycore.pyir import parse_graph_from_ir_without_weight
        g = parse_graph_from_ir_without_weight(ir_txt)
        return g

    def __repr__(self):
        msg = (f"{self.__class__.__name__}({self.name}) has {len(self.nodes)} nodes, "
               f"and has {len(self.input_tensors)} inputs and {len(self.output_tensors)} outputs")
        return msg


def _get_current_batch_size_(self):
    if len(self.nodes) > 0:
        return self.nodes[0].current_batch_size
    else:
        return 1


def _set_current_batch_size_(self, current_batch_size):
    for node in self.nodes:
        node.current_batch_size = current_batch_size


def _get_current_batch_idx_(self):
    if len(self.nodes) > 0:
        return self.nodes[0].current_batch_idx
    else:
        return 0


def _set_current_batch_idx_(self, current_batch_idx):
    for node in self.nodes:
        node.current_batch_idx = current_batch_idx


PyGraph.current_batch_size = property(_get_current_batch_size_, _set_current_batch_size_)
PyGraph.current_batch_idx = property(_get_current_batch_idx_, _set_current_batch_idx_)


class LayerModule(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        for k, v in n.constants.items():
            self.register_parameter(k, torch.nn.Parameter(v.betensor))

    def forward(self):
        self.n.forward()
        # out = []
        # for t in self.n.outputs:
        #     out.append(t.betensor)
        # return out


class GraphModule(torch.nn.Module):
    def __init__(self, input_tensors, output_tensors):
        super().__init__()
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors

    def forward(self, x):
        for i, t in enumerate(self.input_tensors):
            t.betensor = x[i]
        for m in self.children():
            m.forward()
        out = []
        for t in self.output_tensors:
            out.append(t.betensor)
        return out
