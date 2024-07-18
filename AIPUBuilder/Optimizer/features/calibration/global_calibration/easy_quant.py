# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
import torch
import sys


def easy_quant_global_calibration(g, cdataloader, mparams, mscopes):
    vec = mparams
    batches = int(vec[0] if len(vec) > 0 else 1)
    epochs = int(vec[1] if len(vec) > 1 else 1)
    alpha = float(vec[2] if len(vec) > 2 else 0.5)
    beta = float(vec[3] if len(vec) > 3 else 2.0)
    nsteps = int(vec[4] if len(vec) > 4 else 10)
    ngroups = int(vec[5] if len(vec) > 5 else 4)
    msg = (f"easy_quant with batches={batches}, epochs={epochs}, alpha={alpha}, "
           f"beta={beta}, nsteps={nsteps}, ngroups={ngroups}")
    OPT_INFO(msg)
    _easy_quant(g, cdataloader, batches, epochs, alpha, beta, nsteps, ngroups, mscopes)


def _easy_quant(g, cdataloader, batches, epochs, alpha, beta, nsteps, ngroups, mscopes):
    import copy
    from AIPUBuilder.Optimizer.logger import tqdm

    def scale2minmax(scale, is_signed, qrange):
        frange = qrange / scale
        if is_signed:
            return -1 * frange / 2,  frange / 2
        else:
            return 0, frange
    # prevent deleting intermediate tensors
    g.ref_count_tensors = {}
    vdataloader = copy.deepcopy(cdataloader)
    with tqdm(total=batches*epochs*len(g.nodes)*2, desc='easy_quant', file=sys.stdout, leave=True) as pbar:
        for i, sample in enumerate(vdataloader):
            if i >= max(1, batches):  # to save forward times, move batches loop outside instead of calculating mean cos similarity of batches
                break
            inp_data, _ = sample
            for _ in range(epochs):
                g.feed_inputs_data(inp_data)
                # apply scales of last iter
                qg = g.clone()
                g.current_batch_idx = i
                qg.current_batch_idx = i
                g.current_batch_size = vdataloader.batch_size
                qg.current_batch_size = vdataloader.batch_size
                if (i+1) * vdataloader.batch_size > len(vdataloader.dataset):
                    bsize = len(vdataloader.dataset) - i * vdataloader.batch_size
                    g.current_batch_size = bsize
                    qg.current_batch_size = bsize
                qg.clear_tensor_quantization_attrs()
                for n in qg.nodes:
                    if not n.quantized:
                        n.quantize()
                        n.quantized = True
                qg.quantized = True
                # fix Sa, optimize Sw
                for k, n in enumerate(g.nodes):
                    unquantifiable = n.get_param('unquantifiable', optional=True, default_value=False)
                    qn = qg.nodes[k]
                    n.forward()
                    qn.forward()
                    initial_similarity = layer_similarity(n, qn)
                    q_mode_weight = n.attrs["q_mode_weight"]
                    q_mode_activation = n.attrs["q_mode_activation"]
                    if 'weights' in n.constants and not unquantifiable and mscopes.get(n):
                        w = n.constants["weights"]
                        if not w.qinvariant:
                            dn_m = None
                            qrange = 2 ** w.qbits - 1
                            if not QuantMode.is_full_range(q_mode_weight):
                                qrange -= 1
                            if QuantMode.is_per_channel(q_mode_weight):
                                wmins = w.min_key_axis.clone().detach()
                                wmaxs = w.max_key_axis.clone().detach()
                                wscale_chunks = w.scale.chunk(ngroups if ngroups > 0 else w.scale.numel())
                                cstart = 0
                                for chunk in wscale_chunks:
                                    wcs = chunk.float().mean()
                                    max_score = initial_similarity
                                    for s in torch.linspace(alpha * wcs, beta * wcs, nsteps):
                                        wmin, wmax = scale2minmax(s.item(), is_signed(w.dtype), qrange)
                                        dn = n.clone(n.name+"_clone_")
                                        dw = dn.constants["weights"]
                                        dw.min_key_axis[cstart:cstart+chunk.numel()] = wmin
                                        dw.max_key_axis[cstart:cstart+chunk.numel()] = wmax
                                        dn.quantize()
                                        for it, t in enumerate(qn.inputs):
                                            dn.inputs[it].betensor = t.betensor
                                        dn.forward()
                                        score = layer_similarity(n, dn)
                                        if score > max_score:
                                            max_score = score
                                            wmins[cstart:cstart+chunk.numel()] = wmin
                                            wmaxs[cstart:cstart+chunk.numel()] = wmax
                                    cstart += chunk.numel()
                                dn = n.clone(n.name+"_clone_")
                                dw = dn.constants["weights"]
                                dw.min_key_axis = wmins
                                dw.max_key_axis = wmaxs
                                dn.quantize()
                                for it, t in enumerate(qn.inputs):
                                    dn.inputs[it].betensor = t.betensor
                                dn.forward()
                                score = layer_similarity(n, dn)
                                if score > initial_similarity:
                                    dn_m = dn
                                    w.min_key_axis = wmins
                                    w.max_key_axis = wmaxs
                            else:
                                max_score = initial_similarity
                                for s in torch.linspace(alpha * w.scale, beta * w.scale, nsteps):
                                    wmin, wmax = scale2minmax(s.item(), is_signed(w.dtype), qrange)
                                    dn = n.clone(n.name+"_clone_")
                                    dw = dn.constants["weights"]
                                    dw.min = wmin
                                    dw.max = wmax
                                    dn.quantize()
                                    for it, t in enumerate(qn.inputs):
                                        dn.inputs[it].betensor = t.betensor
                                    dn.forward()
                                    score = layer_similarity(n, dn)
                                    if score > max_score:
                                        dn_m = dn
                                        max_score = score
                                        w.min = wmin
                                        w.max = wmax
                            if dn_m:
                                for key in qn.constants.keys():
                                    qn.constants[key] = dn_m.constants[key]
                                for key in qn.params.keys():
                                    qn.params[key] = dn_m.params[key]
                                qn.forward()
                    pbar.update(1)
                # fix Sw, optimize Sa
                tmap = {}
                for k, n in enumerate(g.nodes):
                    unquantifiable = n.get_param('unquantifiable', optional=True, default_value=False)
                    qn = qg.nodes[k]
                    for it, t in enumerate(n.inputs):
                        qn.inputs[it].betensor = linear_quantize_clip(t.betensor, t.scale, t.zerop, t.qmin, t.qmax)
                    qn.forward()
                    initial_similarity = layer_similarity(n, qn)
                    inp_scales = []
                    for idx, inp in enumerate(n.inputs):
                        inp_scales.append((inp.scale, inp.zerop, inp.min, inp.max))
                        if not inp.qinvariant and not unquantifiable and mscopes.get(n):
                            qrange = 2 ** inp.qbits - 1
                            if not QuantMode.is_full_range(q_mode_activation):
                                qrange -= 1
                            max_score = initial_similarity
                            for s in torch.linspace(alpha * inp.scale, beta * inp.scale, nsteps):
                                imin, imax = scale2minmax(s.item(), is_signed(inp.dtype), qrange)
                                dn = n.clone(n.name+"_clone_")
                                dn.inputs[idx].min = imin
                                dn.inputs[idx].max = imax
                                dn.inputs[idx].scale = s.item()
                                dn.inputs[idx].zerop = 0
                                dn.quantize()
                                for it, t in enumerate(n.inputs):
                                    if it == idx:
                                        dn.inputs[it].betensor = linear_quantize_clip(
                                            t.betensor, s.item(), 0, t.qmin, t.qmax)
                                    else:
                                        dn.inputs[it].betensor = linear_quantize_clip(
                                            t.betensor, t.scale, t.zerop, t.qmin, t.qmax)
                                dn.forward()
                                score = layer_similarity(n, dn)
                                if score > max_score:
                                    max_score = score
                                    inp_scales[idx] = (s.item(), 0, imin, imax)
                    if len(inp_scales) > 1:
                        dn = n.clone(n.name+"_clone_")
                        for idx, inp in enumerate(dn.inputs):
                            inp.scale, inp.zerop, inp.min, inp.max = inp_scales[idx]
                            inp.betensor = linear_quantize_clip(
                                n.inputs[idx].betensor, inp.scale, inp.zerop, inp.qmin, inp.qmax)
                        dn.quantize()
                        dn.forward()
                        if layer_similarity(n, dn) <= initial_similarity:
                            for idx, inp in enumerate(n.inputs):
                                inp_scales[idx] = (inp.scale, inp.zerop, inp.min, inp.max)
                    for idx, inp in enumerate(n.inputs):
                        if inp.name not in tmap.keys():
                            tmap[inp.name] = []
                        tmap[inp.name].append(inp_scales[idx])
                    pbar.update(1)
                for k, n in enumerate(g.nodes):
                    for inp in n.inputs:
                        if inp.name in tmap.keys():
                            inp.scale, inp.zerop, inp.min, inp.max = tmap[inp.name][-1]
        pbar.refresh()
