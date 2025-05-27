# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *


def smooth_quant_zy_global_calibration(g, cdataloader, mparams, mscopes):
    vec = mparams
    default_alpha = float(vec[0] if len(vec) > 0 else 0.5)
    auto_tune = bool(vec[1] if len(vec) > 1 else False)
    alpha_min = float(vec[2] if len(vec) > 2 else 0.0)
    alpha_max = float(vec[3] if len(vec) > 3 else 1.0)
    nsteps = int(vec[4] if len(vec) > 4 else 10)
    insert_norm_if_none = bool(vec[5] if len(vec) > 5 else False)
    msg = (f"smooth_quant_zy with default_alpha={default_alpha}, auto_tune={auto_tune}, alpha_min={alpha_min}, "
           f"alpha_max={alpha_max}, nsteps={nsteps}, insert_norm_if_none={insert_norm_if_none}")
    OPT_INFO(msg)
    _smooth_quant_zy(g, cdataloader, default_alpha, auto_tune, alpha_min,
                     alpha_max, nsteps, insert_norm_if_none, mscopes)


def _smooth_quant_zy(g, cdataloader, default_alpha, auto_tune, alpha_min, alpha_max, nsteps, insert_norm_if_none, mscopes):
    from AIPUBuilder.Optimizer.logger import tqdm
    from AIPUBuilder.Optimizer.features import statistic_and_calibration
    import sys
    import copy

    def filter_sigma(sx):
        sx = torch.where(torch.isnan(sx), torch.ones_like(sx), sx)
        sx = torch.where(sx.abs() <= torch.finfo(torch.float32).eps, torch.ones_like(sx), sx)
        sx = torch.where(sx.abs() >= torch.iinfo(torch.int32).max, torch.ones_like(sx), sx)
        return sx

    # find norm-fc pairs
    nfdict = {}
    flist = []
    with tqdm(total=len(g.nodes), desc='smooth_quant_zy: find Norm - FC nodes', file=sys.stdout, leave=True) as pbar:
        for n in g.nodes:
            if n.type in [OpType.RMSNorm, OpType.LayerNorm, OpType.GroupNorm, OpType.InstanceNorm, OpType.BatchNorm, ]:
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
        with tqdm(total=len(g.nodes), desc='smooth_quant_zy: insert BN node for rest FC nodes', file=sys.stdout, leave=True) as pbar:
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

    if auto_tune:
        # prevent deleting intermediate tensors
        g.ref_count_tensors = {}
        vdataloader = copy.deepcopy(cdataloader)
        # forward on batch of samples for searching alpha params
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

    with tqdm(total=len(nfdict), desc='smooth_quant_zy: rebalance', file=sys.stdout, leave=True) as pbar:
        for n, v in nfdict.items():
            skip = False
            for nf in v:
                if not mscopes.get(n):
                    skip = True
            if skip:
                pbar.update(1)
                continue
            best_alpha = default_alpha
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
            sg_nodes = []
            for nc in n.get_descendants()[0]:
                if g.nodes.index(nc) <= last_fc_idx:
                    sg_nodes.append(nc)
            f_inp_max = torch.max(fc_inp_t.max_key_axis.abs().flatten(), fc_inp_t.min_key_axis.abs().flatten())
            f_wgt_max = torch.zeros_like(f_inp_max)
            f_vname_list = []
            gtdict = {}
            for vn in v:
                f_wgt_max = torch.max(f_wgt_max, vn.constants['weights'].betensor.abs().max(0).values.flatten())
                f_vname_list.append(vn.name)
                gtdict[vn.name] = vn.outputs[0].betensor.flatten().float()
            if auto_tune:
                gt = torch.cat([ft for _, ft in sorted(gtdict.items())])
                for alpha in torch.linspace(alpha_min, alpha_max, nsteps):
                    sigma = (f_inp_max ** alpha) / (f_wgt_max ** (1.0-alpha))
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
                    # mse = torch.nn.functional.mse_loss(gt, pt)
                    # sim = torch.nn.functional.cosine_similarity(gt, pt, dim=0)
                    # diff = (gt - pt).abs().max()
                    sqnr = calc_SQNR(gt, pt)
                    # if mse < best_mse:
                    #     best_mse = mse
                    #     best_alpha = alpha
                    # if sim > best_cos:
                    #     best_cos = sim
                    #     best_alpha = alpha
                    # if diff < best_diff:
                    #     best_diff = diff
                    #     best_alpha = alpha
                    if sqnr > best_sqnr:
                        best_sqnr = sqnr
                        best_alpha = alpha
            OPT_DEBUG(f"{n} apply smooth_quant with alpha={best_alpha}")
            sigma = (f_inp_max ** best_alpha) / (f_wgt_max ** (1.0-best_alpha))
            sigma = filter_sigma(sigma)
            n.constants['weights'].betensor /= sigma
            if "biases" in n.constants:
                n.constants['biases'].betensor /= sigma
            for vn in v:
                vn.constants['weights'].betensor *= sigma
            for sn in sg_nodes:
                if sn.name not in f_vname_list:
                    for st in sn.outputs:
                        st.key_axis = fc_inp_t.key_axis
                        st.max_key_axis = fc_inp_t.max_key_axis / sigma
                        st.min_key_axis = fc_inp_t.min_key_axis / sigma
                        st.max = st.max_key_axis.max()
                        st.min = st.min_key_axis.min()
                for _, st in sn.constants.items():
                    statistic_and_calibration(st, sn.attrs, is_constant_tensor=True)
            pbar.update(1)
        pbar.refresh()
