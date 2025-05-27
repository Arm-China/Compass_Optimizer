# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *


def awq_zy_global_calibration(g, cdataloader, mparams, mscopes):
    vec = mparams
    n_grid = int(vec[0] if len(vec) > 0 else 20)
    max_shrink = float(vec[1] if len(vec) > 1 else 0.0)
    insert_norm_if_none = bool(vec[2] if len(vec) > 2 else False)
    msg = (f"awq_zy with n_grid={n_grid}, max_shrink={max_shrink}, insert_norm_if_none={insert_norm_if_none}")
    OPT_INFO(msg)
    _awq_quant_zy(g, cdataloader, n_grid, max_shrink, insert_norm_if_none, mscopes)


def _awq_quant_zy(g, cdataloader, n_grid, max_shrink, insert_norm_if_none, mscopes):
    from AIPUBuilder.Optimizer.logger import tqdm
    from AIPUBuilder.Optimizer.features import statistic_and_calibration
    import sys
    import copy

    def filter_sigma(sx):
        if isinstance(sx, torch.Tensor):
            sx = torch.where(torch.isnan(sx), torch.ones_like(sx), sx)
            sx = torch.where(sx.abs() <= torch.finfo(torch.float32).eps, torch.ones_like(sx), sx)
            sx = torch.where(sx.abs() >= torch.iinfo(torch.int32).max, torch.ones_like(sx), sx)
        return sx

    # find norm-fc pairs and fc-fc pairs
    nfdict = {}
    flist = []
    with tqdm(total=len(g.nodes), desc='awq_zy: find Norm - FC, FC - FC nodes', file=sys.stdout, leave=True) as pbar:
        for n in g.nodes:
            if n.type in [OpType.RMSNorm, OpType.LayerNorm, OpType.GroupNorm, OpType.InstanceNorm, OpType.BatchNorm, OpType.FullyConnected]:
                st = [n, ]
                visited = {n.name: True, }
                while(len(st)):
                    current_node = st.pop()
                    for nchild in current_node.children:
                        if nchild.name not in visited:
                            visited[nchild.name] = True
                            if nchild.type in [OpType.Reshape, OpType.Transpose, OpType.Permute, OpType.Squeeze] and nchild.outputs[0].ir_shape[-1] == n.outputs[0].ir_shape[-1]:
                                st.append(nchild)
                            elif nchild.type in [OpType.FullyConnected, ]:
                                if n not in nfdict:
                                    nfdict[n] = []
                                nfdict[n].append(nchild)
                                flist.append(nchild)
                            else:
                                pass
            pbar.update(1)
        pbar.refresh()
    if insert_norm_if_none:
        inserted_nodes = []
        with tqdm(total=len(g.nodes), desc='awq_zy: insert BN node for rest FC nodes', file=sys.stdout, leave=True) as pbar:
            for n in g.nodes:
                if n.type in [OpType.FullyConnected, ] and n not in flist and mscopes.get(n):
                    bn = PyNode(g.get_valid_node_name(n.name+'_pre_bn'), OpType.BatchNorm)
                    bn_ot = n.inputs[0].clone(g.get_valid_tensor_name(n.name+'_pre_bn_out'))
                    bn.attrs.update(n.attrs.clone())
                    bn.attrs['layer_id'] = '0' + str(n.attrs['layer_id'])
                    bn.params['axis'] = len(bn_ot.ir_shape) - 1
                    bn.constants['weights'] = PyTensor(g.get_valid_tensor_name(
                        n.name+'_pre_bn_weights'), torch.ones(bn_ot.ir_shape[-1], dtype=torch.float32, device=bn_ot.device))
                    bn.constants['biases'] = PyTensor(g.get_valid_tensor_name(
                        n.name+'_pre_bn_biases'), torch.zeros_like(bn.constants['weights'].betensor))
                    bn.add_input(n.inputs[0])
                    bn.add_output(bn_ot)
                    n.remove_input(n.inputs[0])
                    n.add_input(bn_ot)
                    inserted_nodes.append(bn)
                    nfdict[bn] = [n, ]
                    flist.append(n)
                pbar.update(1)
            pbar.refresh()
        for bn in inserted_nodes:
            g.add_node(bn)

    # prevent deleting intermediate tensors
    g.ref_count_tensors = {}
    vdataloader = copy.deepcopy(cdataloader)
    # forward on batch of samples for grid search
    for i, sample in enumerate(vdataloader):
        if i >= 1:
            break
        inp_data, _ = sample
        g.feed_inputs_data(inp_data)
        g.current_batch_idx = i
        if (i+1) * vdataloader.batch_size > len(vdataloader.dataset):
            g.current_batch_size = len(vdataloader.dataset) - i * vdataloader.batch_size
        else:
            g.current_batch_size = vdataloader.batch_size
        for n in g.nodes:
            n.forward()

    with tqdm(total=len(nfdict), desc='awq_zy: rebalance', file=sys.stdout, leave=True) as pbar:
        for n, v in sorted(nfdict.items(), key=lambda x: x[0].attrs['tgid'], reverse=True):
            skip = False
            for nf in v:
                if not mscopes.get(n):
                    skip = True
            if skip:
                pbar.update(1)
                continue
            best_alpha = 0.0
            best_sigma = 1.0
            best_mse = torch.finfo(torch.float32).max
            best_cos = 0.0
            best_diff = torch.finfo(torch.float32).max
            best_sqnr = torch.finfo(torch.float32).min
            last_fc_idx = 0
            for nf in v:
                fidx = g.nodes.index(nf)
                if fidx > last_fc_idx:
                    last_fc_idx = fidx
            fc_inp_t = g.nodes[last_fc_idx].inputs[0].clone()
            # fc_inp_abs_t = g.nodes[last_fc_idx].placeholders[0].clone()
            sg_nodes = []
            for nc in n.get_descendants()[0]:
                if g.nodes.index(nc) <= last_fc_idx:
                    sg_nodes.append(nc)
            # f_inp_abs_mean = fc_inp_abs_t.running_mean_key_axis
            f_inp_abs_mean = fc_inp_t.betensor.abs().view(-1, fc_inp_t.ir_shape[-1]).mean(0)
            f_vname_list = []
            gtdict = {}
            for vn in v:
                f_vname_list.append(vn.name)
                gtdict[vn.name] = vn.outputs[0].betensor.flatten().float()
            gt = torch.cat([ft for _, ft in sorted(gtdict.items())])
            for r in range(n_grid+1):
                alpha = r * 1.0 / n_grid
                sigma = f_inp_abs_mean.pow(alpha).clamp(min=1e-4).view(-1)
                sigma = sigma / (sigma.max() * sigma.min()).sqrt()
                sigma = filter_sigma(sigma)
                sg = g.copy_subgraph(sg_nodes)
                # prevent deleting intermediate tensors
                sg.ref_count_tensors = {}
                for sn in sg.nodes:
                    if sn.name == n.name:
                        sn.constants['weights'].betensor /= sigma
                        if "biases" in sn.constants:
                            sn.constants['biases'].betensor /= sigma
                    if sn.name in f_vname_list:
                        sn.constants['weights'].betensor *= sigma
                # sg.nodes' input betensors are all existed and copyed as we have forwarded one batch samples on g
                for sn in sg.nodes:
                    if sn.name not in f_vname_list:
                        for st in sn.outputs:
                            st.key_axis = fc_inp_t.key_axis
                            st.max_key_axis = fc_inp_t.max_key_axis / sigma
                            st.min_key_axis = fc_inp_t.min_key_axis / sigma
                            st.max = st.max_key_axis.max()
                            st.min = st.min_key_axis.min()
                    for _, st in sn.constants.items():
                        statistic_and_calibration(st, sn.attrs, is_constant_tensor=True)
                    sn.params['unquantifiable'] = False
                    sn.quantize()
                ptdict = {}
                for sn in sg.nodes:
                    sn.forward()
                    if sn.name in f_vname_list:
                        ptdict[sn.name] = linear_dequantize(
                            sn.outputs[0].betensor, sn.outputs[0].scale, sn.outputs[0].zerop).flatten().float()
                pt = torch.cat([ft for _, ft in sorted(ptdict.items())])
                mse = torch.nn.functional.mse_loss(gt, pt)
                if mse < best_mse:
                    best_mse = mse
                    best_sigma = sigma
                    best_alpha = alpha

            OPT_DEBUG(f"{n} apply awq with alpha={best_alpha}")
            n.constants['weights'].betensor /= best_sigma
            if "biases" in n.constants:
                n.constants['biases'].betensor /= best_sigma
            for vn in v:
                vn.constants['weights'].betensor *= best_sigma
            for sn in sg_nodes:
                if sn.name not in f_vname_list:
                    for st in sn.outputs:
                        st.key_axis = fc_inp_t.key_axis
                        st.max_key_axis = fc_inp_t.max_key_axis / best_sigma
                        st.min_key_axis = fc_inp_t.min_key_axis / best_sigma
                        st.max = st.max_key_axis.max()
                        st.min = st.min_key_axis.min()
                for _, st in sn.constants.items():
                    statistic_and_calibration(st, sn.attrs, is_constant_tensor=True)

            # search for clip shrink ratio
            max_shrink = min(max(max_shrink, 0.0), 1.0)
            best_ratio = 1.0
            best_mse = torch.finfo(torch.float32).max
            for r in range(int(max_shrink * n_grid)):
                ratio = (1.0 - r * 1.0 / n_grid)
                sg = g.copy_subgraph(sg_nodes)
                # prevent deleting intermediate tensors
                sg.ref_count_tensors = {}
                for sn in sg.nodes:
                    if sn.name in f_vname_list:
                        w = sn.constants['weights']
                        w.max *= ratio
                        w.min *= ratio
                        if w.max_key_axis is not None:
                            w.max_key_axis *= ratio
                        if w.min_key_axis is not None:
                            w.min_key_axis *= ratio
                    sn.params['unquantifiable'] = False
                    sn.quantize()
                ptdict = {}
                for sn in sg.nodes:
                    sn.forward()
                    if sn.name in f_vname_list:
                        ptdict[sn.name] = linear_dequantize(
                            sn.outputs[0].betensor, sn.outputs[0].scale, sn.outputs[0].zerop).flatten().float()
                pt = torch.cat([ft for _, ft in sorted(ptdict.items())])
                mse = torch.nn.functional.mse_loss(gt, pt)
                if mse < best_mse:
                    best_mse = mse
                    best_ratio = ratio
            OPT_DEBUG(f"{n} apply awq clipping with ratio={best_ratio}")
            for vn in v:
                w = vn.constants['weights']
                w.max *= best_ratio
                w.min *= best_ratio
                if w.max_key_axis is not None:
                    w.max_key_axis *= best_ratio
                if w.min_key_axis is not None:
                    w.min_key_axis *= best_ratio
            pbar.update(1)
        pbar.refresh()
