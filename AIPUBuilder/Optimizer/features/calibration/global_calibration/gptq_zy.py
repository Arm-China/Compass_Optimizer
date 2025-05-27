# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_INFO, OPT_DEBUG, OPT_WARN
import torch
import sys
import math
import copy


def gptq_zy_global_calibration(g, cdataloader, mparams, mscopes):
    vec = mparams
    batches = int(vec[0] if len(vec) > 0 else 1)
    # whether to apply the activation order GPTQ heuristic
    use_act_order = bool(vec[1] if len(vec) > 1 else True)
    # percent of the average Hessian diagonal to use for dampening
    perc_damp = float(vec[2] if len(vec) > 2 else 0.01)
    block_size = int(vec[3] if len(vec) > 3 else 128)

    msg = (
        f"gptq_zy with batches={batches}, use_act_order={use_act_order}, perc_damp={perc_damp}, block_size={block_size}")
    OPT_INFO(msg)
    _gptq_zy(g, cdataloader, batches, use_act_order, perc_damp, block_size, mscopes)


def _gptq_zy(g, cdataloader, batches, use_act_order, perc_damp, block_size, mscopes):
    Hdict = {}
    vdataloader = copy.deepcopy(cdataloader)
    with tqdm(total=batches*len(g.nodes), desc='gptq_zy: compute Hessian matrix', file=sys.stdout, consumer=g) as pbar:
        for i, sample in enumerate(vdataloader):
            if i >= max(1, batches):
                break
            g.current_batch_idx = i
            if (i+1) * vdataloader.batch_size > len(vdataloader.dataset):
                g.current_batch_size = len(vdataloader.dataset) - i * vdataloader.batch_size
            else:
                g.current_batch_size = vdataloader.batch_size
            cur_samples = i * vdataloader.batch_size
            inp_data, _ = sample
            g.feed_inputs_data(inp_data)
            g.reset_edge_tensors_ref_count()
            for k, n in enumerate(g.nodes):
                if n.type in [OpType.FullyConnected, OpType.DepthwiseConv, OpType.Convolution] and mscopes.get(n):
                    inp = n.inputs[0].betensor.clone()
                    w = n.constants['weights'].betensor
                    if len(inp.shape) == 2:
                        inp = inp.unsqueeze(0)
                    if OpType.FullyConnected == n.type:
                        inp = inp.reshape((-1, inp.shape[-1]))
                        inp = inp.t()
                    else:
                        inp = nhwc2nchw(inp)
                        inp = torch.nn.functional.pad(inp, (n.get_param('pad_left'), n.get_param(
                            'pad_right'), n.get_param('pad_top'), n.get_param('pad_bottom')))
                        unfold_func = torch.nn.Unfold((n.get_param("kernel_y"), n.get_param("kernel_x")),
                                                      dilation=(n.get_param('dilation_y'), n.get_param('dilation_x')),
                                                      padding=0,
                                                      stride=(n.get_param("stride_y"), n.get_param("stride_x"))
                                                      )
                        inp = unfold_func(inp)
                        inp = inp.permute([1, 0, 2])
                        inp = inp.flatten(1)
                    Hcolumns = inp.shape[0]
                    if n not in Hdict:
                        Hdict[n] = torch.zeros(Hcolumns, Hcolumns, device=w.device)
                    H = Hdict[n].to(w.device)
                    H *= cur_samples / (cur_samples + g.current_batch_size)
                    inp = math.sqrt(2.0 / (cur_samples + g.current_batch_size)) * inp.float()
                    H += inp.matmul(inp.t())
                    Hdict[n] = H.cpu()
                n.forward()
                pbar.update(1)
            tz = PyTensor('null').betensor
            for n in g.nodes:
                for pld in n.placeholders:
                    del pld.betensor
                    pld.betensor = tz
                for t in n.outputs:
                    if t not in g.output_tensors:
                        del t.betensor
                        t.betensor = tz
            g.reset_edge_tensors_ref_count()
        pbar.refresh()
    with tqdm(total=len(Hdict), desc='gptq_zy: quantize weights', file=sys.stdout, leave=True) as pbar:
        ll = 0
        for n, v in Hdict.items():
            w_scale, w_zerop, w_qmin, w_qmax, w_dtype = get_linear_quant_params_from_tensor(n.constants['weights'],
                                                                                            n.attrs["q_mode_weight"],
                                                                                            n.attrs["q_bits_weight"],
                                                                                            is_signed=True)
            w = n.constants['weights'].betensor.clone().float()
            if OpType.FullyConnected == n.type:
                pass
            else:
                w = w.flatten(1)
            H = v.to(w.device)
            Hcolumns = H.shape[1]
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            w[:, dead] = 0

            if use_act_order:
                perm = torch.argsort(torch.diag(H), descending=True)
                w = w[:, perm]
                H = H[perm][:, perm]
                invperm = torch.argsort(perm)
            losses = torch.zeros_like(w)
            qw = torch.zeros_like(w)
            damp = perc_damp * torch.mean(torch.diag(H))
            diag = torch.arange(Hcolumns, device=w.device)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H
            for i1 in range(0, Hcolumns, block_size):
                i2 = min(i1 + block_size, Hcolumns)
                count = i2 - i1
                w1 = w[:, i1:i2].clone()
                qw1 = torch.zeros_like(w1)
                err1 = torch.zeros_like(w1)
                losses1 = torch.zeros_like(w1)
                Hinv1 = Hinv[i1:i2, i1:i2]
                for i in range(count):
                    w2 = w1[:, i]
                    d = Hinv1[i, i]
                    w2_scale = w_scale
                    w2_zerop = w_zerop
                    qw2 = linear_dequantize(linear_quantize_clip(w2, w2_scale, w2_zerop, w_qmin,
                                            w_qmax), w2_scale, w2_zerop).flatten().to(w2.dtype)
                    qw1[:, i] = qw2
                    losses1[:, i] = (w2 - qw2) ** 2 / d ** 2
                    err2 = (w2 - qw2) / d
                    w1[:, i:] -= err2.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    err1[:, i] = err2
                qw[:, i1:i2] = qw1
                losses[:, i1:i2] = losses1 / 2
                w[:, i2:] -= err1.matmul(Hinv[i1:i2, i2:])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if use_act_order:
                qw = qw[:, invperm]
            n.attrs['gptq_weights'] = {n.attrs['q_bits_weight']: qw.reshape(n.constants['weights'].betensor.shape)}
            pbar.update(1)
            ll += 1
        pbar.refresh()
