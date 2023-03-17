# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_INFO, OPT_DEBUG
import torch
import sys


def adaround_global_calibration(g, cdataloader, cstrategy):
    from AIPUBuilder.Optimizer.config import GlobalCalibrationParamField
    valid, vec = GlobalCalibrationParamField._parse(cstrategy)
    if valid:
        batches = int(vec[0] if len(vec) > 0 else 1)
        epochs = int(vec[1] if len(vec) > 1 else 1)
        batch_size = int(vec[2] if len(vec) > 2 else 1)
        lrate = float(vec[3] if len(vec) > 3 else 0.001)
        reg_param = float(vec[4] if len(vec) > 4 else 0.01)
        beta_start = float(vec[5] if len(vec) > 5 else 20)
        beta_end = float(vec[6] if len(vec) > 6 else 2)
        warm_start = float(vec[7] if len(vec) > 7 else 0.2)

        msg = (f"adaround with batches={batches}, epochs={epochs}, batch_size={batch_size}, lr={lrate}, "
               f"reg_param={reg_param}, beta_start={beta_start}, beta_end={beta_end}, warm_start={warm_start}")
        OPT_INFO(msg)
        _adaround(g, cdataloader, batches, epochs, batch_size, lrate, reg_param, beta_start, beta_end, warm_start)


def _adaround(g, cdataloader, batches, epochs, batch_size, lrate, reg_param, beta_start, beta_end, warm_start):
    class QNodeModule (torch.nn.Module):
        def __init__(self, n, qn):
            super().__init__()
            self.n = n.clone(n.name + "_clone_")
            self.gamma, self.zeta = -0.1, 1.1
            # init alpha
            wf = n.constants['weights'].betensor
            wscale = qn.constants['weights'].scale
            wzerop = qn.constants['weights'].zerop
            wqmin = qn.constants['weights'].qmin
            wqmax = qn.constants['weights'].qmax
            if isinstance(wscale, torch.Tensor):
                # expand scale/zerop shape to match weights
                max_cnt = 0
                mshape = None
                for d in range(wf.dim()):
                    if wf.shape[d] == wscale.numel():
                        sz_shape = [1] * wf.dim()
                        sz_shape[d] = -1
                        wq = linear_quantize_clip(wf, wscale.reshape(sz_shape), wzerop.reshape(sz_shape), wqmin, wqmax)
                        cnt = wq.numel() - (wq - qn.constants['weights'].betensor).count_nonzero()
                        if cnt >= max_cnt:
                            max_cnt = cnt
                            mshape = sz_shape
                wscale = wscale.reshape(mshape)
                wzerop = wzerop.reshape(mshape)
            wqf = wf * wscale
            rest = wqf - wqf.floor()
            alpha0 = - torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1.)
            self.alpha = torch.nn.Parameter(alpha0, requires_grad=True)
            self.wscale = wscale
            self.wzerop = wzerop
            self.wqmin = wqmin
            self.wqmax = wqmax

        def forward(self, x):
            rest = self.calc_h_alpha()
            # quantize then dequantize weights
            wqf = torch.floor(self.n.constants['weights'].betensor * self.wscale) + rest
            wf = linear_dequantize(torch.clamp(wqf - self.wzerop, self.wqmin, self.wqmax), self.wscale, self.wzerop)
            orginal_wf = self.n.constants['weights'].betensor
            self.n.constants['weights'].betensor = wf
            for i, t in enumerate(self.n.inputs):
                t.betensor = x[i]
            out = self.n.forward()
            self.n.constants['weights'].betensor = orginal_wf
            return out

        def get_optimized_weights(self):
            return torch.clamp(torch.floor(self.n.constants['weights'].betensor * self.wscale) + (self.alpha >= 0).float() - self.wzerop, self.wqmin, self.wqmax)

        def calc_h_alpha(self):
            return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0., 1.)

        def reconstruction_loss(self, gt, pt):
            return (pt - gt).abs().pow(2.0).sum(-1).mean()  # NHWC

        def regularization_term(self, reg_param, beta_start, beta_end, warm_start, num_iter, cur_iter):
            warm_start_end_iter = warm_start * num_iter
            if cur_iter < warm_start_end_iter:
                return 0.
            else:
                import math
                h_alpha = self.calc_h_alpha()
                iter_ratio = (cur_iter - warm_start_end_iter) * 1.0 / (num_iter - warm_start_end_iter)
                beta = beta_end + 0.5 * (beta_start - beta_end) * (1 + math.cos(iter_ratio * math.pi))
                return (1.0 - (2 * h_alpha - 1).abs().pow(beta)).sum() * reg_param

    qg = g.clone()
    qg.clear_tensor_quantization_attrs()
    for n in qg.nodes:
        if not n.quantized:
            n.quantize()
            n.quantized = True
    qg.quantized = True

    import copy
    import math
    from AIPUBuilder.Optimizer.logger import tqdm
    vdataloader = copy.deepcopy(cdataloader)
    cached_float_tensors = {}
    # collect all inputs tensors into cached dict firstly
    for i, sample in enumerate(vdataloader):
        if i >= max(1, batches):
            break
        inp_data, _ = sample
        g.feed_inputs_data(inp_data)
        for inp in g.input_tensors:
            t = inp.betensor.clone().detach()
            if inp.name not in cached_float_tensors.keys():
                cached_float_tensors[inp.name] = t
            else:
                cached_float_tensors[inp.name] = torch.cat((cached_float_tensors[inp.name], t), dim=0)
    sample_num = 0
    for key, val in cached_float_tensors.items():
        psize = int(math.ceil(val.shape[0] * 1.0 / batch_size) * batch_size - val.shape[0])
        if psize < val.shape[0]:
            cached_float_tensors[key] = torch.cat((val, val[:psize]), dim=0)
        else:
            t = torch.cat((val, val), dim=0)
            while t.shape[0] < batch_size:
                t = torch.cat((t, t), dim=0)
            cached_float_tensors[key] = t[:batch_size]
        sample_num = cached_float_tensors[key].shape[0]
    # count each tensor's reference count
    ref_count_float_tensors = {}
    for n in g.nodes:
        for inp in n.inputs:
            if inp.name not in ref_count_float_tensors.keys():
                ref_count_float_tensors[inp.name] = 1
            else:
                ref_count_float_tensors[inp.name] += 1
    # optimize each node
    iterations = sample_num // batch_size
    abnormal_tensors = {}
    for it in g.input_tensors:
        it.betensor = cached_float_tensors[it.name].to(it.betensor.device)
    for it in qg.input_tensors:
        it.betensor = cached_float_tensors[it.name].to(it.betensor.device)
    cached_float_tensors = {}
    cached_quant_tensors = {}
    with tqdm(total=iterations*epochs*len(g.nodes), desc='adaround', file=sys.stdout, leave=True) as pbar:
        for k, n in enumerate(g.nodes):
            qn = qg.nodes[k]
            # move inputs to device
            for it in n.inputs:
                cached_float_tensors[it.name] = cached_float_tensors[it.name].to(it.betensor.device)
                cached_quant_tensors[it.name] = cached_quant_tensors[it.name].to(it.betensor.device)
            # forward featuremaps on float graph
            for it in n.inputs:
                it.betensor = cached_float_tensors[it.name]
            n.current_batch_size = sample_num
            n.current_batch_idx = 0
            n.forward()
            for ot in n.outputs:
                if len(ot.betensor.shape) < 1 or ot.betensor.shape[0] != sample_num:
                    # no batch dim
                    abnormal_tensors[ot.name] = True
                if ot.name not in cached_float_tensors.keys():
                    cached_float_tensors[ot.name] = ot.betensor
            # apply adaround on current layer
            unquantifiable = n.get_param('unquantifiable', optional=True, default_value=False)
            if 'weights' in n.constants and not unquantifiable and n.type not in [OpType.GRUv3, OpType.GRUv1]:
                qmodule = QNodeModule(n, qn)
                optim = torch.optim.Adam([qmodule.alpha], lr=lrate)
                cur_iter = 0
                for _ in range(epochs):
                    cur_rand_indices = torch.randperm(sample_num)
                    cur_batch_idx = 0
                    for i in range(iterations):
                        optim.zero_grad()
                        cur_float_inp = []
                        cur_float_out = []
                        indices = cur_rand_indices[i*batch_size: (i+1)*batch_size]
                        for it in n.inputs:
                            t = cached_float_tensors[it.name] if it.name in abnormal_tensors.keys(
                            ) else cached_float_tensors[it.name][indices]
                            cur_float_inp.append(t)
                        for ot in n.outputs:
                            t = cached_float_tensors[ot.name] if ot.name in abnormal_tensors.keys(
                            ) else cached_float_tensors[ot.name][indices]
                            cur_float_out.append(t)
                        cur_qmodule_inp = []
                        for it in qn.inputs:
                            t = cached_quant_tensors[it.name] if it.name in abnormal_tensors.keys(
                            ) else cached_quant_tensors[it.name][indices]
                            t = linear_dequantize(t, it.scale, it.zerop)
                            cur_qmodule_inp.append(t)
                        qmodule.n.current_batch_idx = cur_batch_idx
                        qmodule.n.current_batch_size = batch_size
                        qmodule.forward(cur_qmodule_inp)

                        def get_act_func_ret(n, idx, x):
                            for chl in n.children:
                                if n.outputs[idx] in chl.inputs and OpType.Activation == chl.type:
                                    chl.inputs[0].betensor = x
                                    chl.current_batch_idx = cur_batch_idx
                                    chl.current_batch_size = batch_size
                                    chl.forward()
                                    y = chl.outputs[0].betensor
                                    return y
                            return x
                        cur_qmodule_out = []
                        for idx, ot in enumerate(qmodule.n.outputs):
                            yp = get_act_func_ret(n, idx, ot.betensor)
                            cur_qmodule_out.append(yp)
                            yg = get_act_func_ret(n, idx, cur_float_out[idx])
                            cur_float_out[idx] = yg
                        if len(cur_qmodule_out) < 2:
                            pt = cur_qmodule_out[0]
                        else:
                            pt = torch.cat([t.flatten() for t in cur_qmodule_out])
                        if len(cur_float_out) < 2:
                            gt = cur_float_out[0]
                        else:
                            gt = torch.cat([t.flatten() for t in cur_float_out])
                        recon_loss = qmodule.reconstruction_loss(gt, pt)
                        round_loss = qmodule.regularization_term(
                            reg_param, beta_start, beta_end, warm_start, iterations*epochs, cur_iter)
                        total_loss = recon_loss + round_loss
                        total_loss.backward()
                        optim.step()
                        if 0 == cur_iter % 100:
                            OPT_DEBUG("Adaround optimization of layer_id=%s, %s, %s\n iterations=%d, loss=%.5f, recon_loss=%.5f, round_loss=%.5f" % (
                                str(n.attrs['layer_id']), str(n.type), n.name, cur_iter, float(total_loss), float(recon_loss), float(round_loss)))
                        cur_iter += 1
                        cur_batch_idx += 1
                        pbar.update(1)
                qnw = qn.constants['weights']
                qnw.betensor = qmodule.get_optimized_weights()
                # record adaround weights to source node
                n.attrs['adaround_weights'] = {qn.attrs['q_bits_weight']
                    : linear_dequantize(qnw.betensor, qmodule.wscale, qmodule.wzerop)}
            else:
                pbar.update(iterations*epochs)
            # forward featuremaps on quant graph
            for it in qn.inputs:
                it.betensor = cached_quant_tensors[it.name]
            qn.current_batch_size = sample_num
            qn.current_batch_idx = 0
            qn.forward()
            for ot in qn.outputs:
                if ot.name not in cached_quant_tensors.keys():
                    cached_quant_tensors[ot.name] = ot.betensor
            # reduce tensor's reference count
            for it in n.inputs:
                ref_count_float_tensors[it.name] -= 1
            # clear useless tensors out of cache for memory saving
            useless_tnames = []
            for rkey, rval in ref_count_float_tensors.items():
                if rval < 1:
                    useless_tnames.append(rkey)
            for rkey in useless_tnames:
                if rkey in cached_float_tensors.keys():
                    rval = cached_float_tensors.pop(rkey)
                    del rval
                if rkey in cached_quant_tensors.keys():
                    rval = cached_quant_tensors.pop(rkey)
                    del rval
            # move tensors back to cpu
            for t in n.inputs:
                if t.name in cached_float_tensors.keys():
                    cached_float_tensors[t.name] = cached_float_tensors[t.name].cpu()
                if t.name in cached_quant_tensors.keys():
                    cached_quant_tensors[t.name] = cached_quant_tensors[t.name].cpu()
            for t in n.outputs:
                if t.name in cached_float_tensors.keys():
                    cached_float_tensors[t.name] = cached_float_tensors[t.name].cpu()
                if t.name in cached_quant_tensors.keys():
                    cached_quant_tensors[t.name] = cached_quant_tensors[t.name].cpu()

            def reset_layer_tensors(n):
                for t in n.inputs:
                    ss = None
                    try:
                        ss = list(t.ir_shape)
                    except:
                        ss = list(t.shape)
                    t.betensor = torch.zeros(ss, device=t.betensor.device)
                for t in n.outputs:
                    ss = None
                    try:
                        ss = list(t.ir_shape)
                    except:
                        ss = list(t.shape)
                    t.betensor = torch.zeros(ss, device=t.betensor.device)
                for t in n.placeholders:
                    ss = None
                    try:
                        ss = list(t.ir_shape)
                    except:
                        ss = list(t.shape)
                    t.betensor = torch.zeros(ss, device=t.betensor.device)
            reset_layer_tensors(n)
            reset_layer_tensors(qn)
