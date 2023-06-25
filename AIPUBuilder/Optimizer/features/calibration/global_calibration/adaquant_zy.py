# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_INFO, OPT_DEBUG, OPT_WARN
import torch
import sys


def adaquant_zy_global_calibration(g, cdataloader, mparams, mscopes):
    vec = mparams
    batches = int(vec[0] if len(vec) > 0 else 1)
    epochs = int(vec[1] if len(vec) > 1 else 1)
    batch_size = int(vec[2] if len(vec) > 2 else 1)
    lr_w = float(vec[3] if len(vec) > 5 else 0.00001)
    lr_b = float(vec[4] if len(vec) > 6 else 0.001)
    lr_qpw = float(vec[5] if len(vec) > 4 else 0.001)
    lr_qpa = float(vec[6] if len(vec) > 3 else 0.1)
    msg = (f"adaquant_zy with batches={batches}, epochs={epochs}, batch_size={batch_size}, "
           f"lr_weight={lr_w}, lr_bias={lr_b}, lr_qp_wht={lr_qpw}, lr_qp_act={lr_qpa}")
    OPT_INFO(msg)
    _adaquant_zy(g, cdataloader, batches, epochs, batch_size, lr_w, lr_b, lr_qpw, lr_qpa, mscopes)


def _adaquant_zy(g, cdataloader, batches, epochs, batch_size, lr_w, lr_b, lr_qpw, lr_qpa, mscopes):
    def is_in_scopes(layer_n):
        if len(mscopes) < 1:
            return True
        else:
            return layer_n.type.name.lower() in mscopes or int(layer_n.attrs['layer_id']) in mscopes

    class QNodeModule (torch.nn.Module):
        def __init__(self, n, qn, lr_w, lr_b, lr_qpw, lr_qpa, only_optim_inp):
            super().__init__()
            dev = n.outputs[0].betensor.device
            self.n = n.clone(n.name + "_clone_")

            def parameterize(t, optim_d, optim_s):
                it = PyTensor('tmp')
                s = 1.0
                z = 0
                if isinstance(t.scale, torch.Tensor):
                    kshape = [t.shape[ax] if ax == t.key_axis_c else 1 for ax in range(len(t.shape))]
                    s = t.scale.float().clone().detach().reshape(kshape)
                else:
                    s = torch.tensor(t.scale, dtype=torch.float, device=dev)
                if isinstance(t.zerop, torch.Tensor):
                    kshape = [t.shape[ax] if ax == t.key_axis_c else 1 for ax in range(len(t.shape))]
                    z = t.zerop.float().clone().detach().reshape(kshape)
                else:
                    z = torch.tensor(t.zerop, dtype=torch.float, device=dev)
                it.scale = torch.nn.Parameter(s, requires_grad=True) if optim_s else s
                it.zerop = z
                it.qmin = t.qmin
                it.qmax = t.qmax
                if optim_d:
                    it.betensor = torch.nn.Parameter(torch.zeros_like(
                        t.betensor, dtype=torch.float), requires_grad=True)
                return it

            self.inputs = []
            for t in qn.inputs:
                self.inputs.append(parameterize(t, optim_d=False, optim_s=True))
            self.constants = {}
            for key, t in qn.constants.items():
                if key in n.constants:
                    self.constants[key] = parameterize(t, optim_d=True, optim_s=True)

            self.optim_qpw = None
            self.optim_w = None
            if ('weights' in self.constants) and not only_optim_inp:
                self.optim_w = torch.optim.Adam([self.constants['weights'].betensor], lr=lr_w)
                self.optim_qpw = torch.optim.Adam([self.constants['weights'].scale], lr=lr_qpw)
            self.optim_b = None
            if ('biases' in self.constants) and not only_optim_inp:
                self.optim_b = torch.optim.Adam([self.constants['biases'].betensor], lr=lr_b)
            self.optim_qpa = None
            if len(self.inputs) > 0:
                self.optim_qpa = torch.optim.Adam([it.scale for it in self.inputs], lr=lr_qpa)

        def forward(self, x):
            bak_constants = {}
            for k, v in self.constants.items():
                bak_constants[k] = self.n.constants[k].betensor
                self.n.constants[k].betensor
            # quantize then dequantize weights
            if 'weights' in self.n.constants and self.optim_w:
                # linear_quantize_clip will break the backward (round cause 0 grad)
                self.n.constants['weights'].betensor = self.get_optimized_weights()
            if 'biases' in self.n.constants and self.optim_b:
                self.n.constants['biases'].betensor = self.get_optimized_biases()
            # quantize then dequantize input
            for i, t in enumerate(self.n.inputs):
                ip = self.inputs[i]
                xi = linear_quantize_clip(x[i], ip.scale, ip.zerop, ip.qmin, ip.qmax)
                t.betensor = linear_dequantize(xi, ip.scale, ip.zerop)

            out = self.n.forward()

            for k, _ in self.constants.items():
                self.n.constants[k].betensor = bak_constants[k]
            return out

        def zero_grad(self):
            if self.optim_w:
                self.optim_w.zero_grad()
            if self.optim_b:
                self.optim_b.zero_grad()
            if self.optim_qpw:
                self.optim_qpw.zero_grad()
            if self.optim_qpa:
                self.optim_qpa.zero_grad()

        def step(self):
            if self.optim_w:
                self.optim_w.step()
            if self.optim_b:
                self.optim_b.step()
            if self.optim_qpw:
                self.optim_qpw.step()
            if self.optim_qpa:
                self.optim_qpa.step()

        def get_optimized_weights(self):
            wf = self.n.constants['weights'].betensor
            wp = self.constants['weights']
            wqf = linear_quantize_clip(wf, wp.scale, wp.zerop, wp.qmin, wp.qmax) + wp.betensor
            wf = linear_dequantize(wqf, wp.scale, wp.zerop)
            return wf

        def get_optimized_biases(self):
            bf = self.n.constants['biases'].betensor
            bp = self.constants['biases']
            bs = self.inputs[0].scale
            if 'weights' in self.n.constants:
                bs = self.inputs[0].scale * self.constants['weights'].scale
            if isinstance(bs, torch.Tensor):
                bs = torch.squeeze(bs)
            bqf = linear_quantize_clip(bf, bs, 0, bp.qmin, bp.qmax) + bp.betensor
            bf = linear_dequantize(bqf, bs, 0)
            return bf

        def get_optimized_weights_range(self):
            wp = self.constants['weights']
            fmin = linear_dequantize(wp.qmin, wp.scale, wp.zerop).flatten()
            fmax = linear_dequantize(wp.qmax, wp.scale, wp.zerop).flatten()
            if fmin.numel() > 1:
                return fmin, fmax
            else:
                return fmin.item(), fmax.item()

        def get_optimized_inp_range(self, i):
            ip = self.inputs[i]
            fmin = linear_dequantize(ip.qmin, ip.scale, ip.zerop).flatten()
            fmax = linear_dequantize(ip.qmax, ip.scale, ip.zerop).flatten()
            if fmin.numel() > 1:
                return fmin, fmax
            else:
                return fmin.item(), fmax.item()
    # prevent deleting intermediate tensors
    g.ref_count_tensors = {}

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
    with tqdm(total=iterations*epochs*len(g.nodes), desc='adaquant_zy', file=sys.stdout, leave=True) as pbar:
        for k, n in enumerate(g.nodes):
            qn = qg.nodes[k]
            # move inputs to device
            for it in n.inputs:
                cached_float_tensors[it.name] = cached_float_tensors[it.name].to(it.betensor.device)
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
                    OPT_WARN(f"{n.name} type={n.type} layer_id={n.attrs['layer_id']} batch dim is abnormal: expect batch_dim=0 and batches={sample_num}, but got shape={ot.betensor.shape}."
                             f"you may try to set batch_size = calibration_batch_size x batches or batch_size=calibration_batch_size=batches=1", log_once=True)
                if ot.name not in cached_float_tensors.keys():
                    cached_float_tensors[ot.name] = ot.betensor
            # apply adaround on current layer
            unquantifiable = n.get_param('unquantifiable', optional=True, default_value=False)
            if n.type != OpType.Input and not unquantifiable and is_in_scopes(n):
                only_optim_inp = True
                if 'weights' in n.constants and n.type not in [OpType.GRUv3, OpType.GRUv1]:
                    only_optim_inp = False
                qmodule = QNodeModule(n, qn, lr_w, lr_b, lr_qpw, lr_qpa, only_optim_inp)
                cur_iter = 0
                scheduler_qpa = torch.optim.lr_scheduler.CosineAnnealingLR(
                    qmodule.optim_qpa, epochs*iterations) if qmodule.optim_qpa else None
                for _ in range(epochs):
                    cur_rand_indices = torch.randperm(sample_num)
                    cur_batch_idx = 0
                    for i in range(iterations):
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
                        qmodule.n.current_batch_idx = cur_batch_idx
                        qmodule.n.current_batch_size = batch_size
                        qmodule.forward(cur_float_inp)

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
                        loss = torch.nn.functional.mse_loss(pt.double(), gt.double())
                        qmodule.zero_grad()
                        try:
                            loss.backward()
                        except:
                            OPT_DEBUG("Adaquant optimization of layer_id=%s, %s, %s\n does not support backward." % (
                                str(n.attrs['layer_id']), str(n.type), n.name))
                        qmodule.step()
                        if 0 == cur_iter % 100:
                            OPT_DEBUG("Adaquant optimization of layer_id=%s, %s, %s\n iterations=%d, loss=%.5f " % (
                                str(n.attrs['layer_id']), str(n.type), n.name, cur_iter, float(loss)))
                        cur_iter += 1
                        cur_batch_idx += 1
                        pbar.update(1)
                        if scheduler_qpa:
                            scheduler_qpa.step()
                            # print(qmodule.optim_qpa.param_groups[0]["lr"])
                for ii, ti in enumerate(n.inputs):
                    ti.min, ti.max = qmodule.get_optimized_inp_range(ii)
                if not only_optim_inp:
                    tw = n.constants['weights']
                    wmin, wmax = qmodule.get_optimized_weights_range()
                    if isinstance(wmin, torch.Tensor):
                        tw.min_key_axis, tw.max_key_axis = wmin, wmax
                    else:
                        tw.min, tw.max = wmin, wmax
                    # record adaround weights to source node
                    n.attrs['adaquant_weights'] = {
                        qn.attrs['q_bits_weight']: qmodule.get_optimized_weights().clone().detach()}
                    if 'biases' in n.constants:
                        n.attrs['adaquant_biases'] = {
                            qn.attrs['q_bits_bias']: qmodule.get_optimized_biases().clone().detach()}
            else:
                pbar.update(iterations*epochs)

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
            # move tensors back to cpu
            for t in n.inputs:
                if t.name in cached_float_tensors.keys():
                    cached_float_tensors[t.name] = cached_float_tensors[t.name].cpu()
            for t in n.outputs:
                if t.name in cached_float_tensors.keys():
                    cached_float_tensors[t.name] = cached_float_tensors[t.name].cpu()

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
