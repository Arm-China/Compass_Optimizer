# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import tqdm, OPT_ERROR
import torch
import sys


def svd_based_quant_global_calibration(g, cdataloader, mparams, mscopes):
    vec = mparams
    mode = float(vec[0] if len(vec) > 0 else 0)
    alpha = float(vec[1] if len(vec) > 1 else 0.5)
    beta = float(vec[2] if len(vec) > 2 else 2.0)
    nsteps = int(vec[3] if len(vec) > 3 else 10)
    thresh = float(vec[4] if len(vec) > 4 else 0.09)
    msg = (
        f"svd_based_search_scale with mode={int(mode)} alpha={alpha}, "f"beta={beta}, nsteps={nsteps}, thresh={thresh}")
    OPT_INFO(msg)
    oplist = (set(), set(), set())
    if int(mode) <= 0:
        oplist = (mscopes, set(), set())
    elif int(mode) == 1:
        oplist = (set(), mscopes, set())
    elif int(mode) >= 2:
        oplist = (set(), set(), mscopes)
    _svd_based_search_scale(g, cdataloader, alpha, beta, nsteps, thresh, mode, oplist)


def _svd_based_search_scale(g, cdataloader, alpha, beta, nsteps, thresh, mode, oplist):
    import copy

    def get_qmin_qmax(outsign, qbits):
        q_max, q_min = 2 ** qbits - 1, 0
        if outsign:
            if QuantMode.is_full_range(quant_mode):
                q_max = 2 ** (qbits - 1) - 1
                q_min = -1 * q_max - 1
            else:
                q_max = 2 ** (qbits - 1) - 1
                q_min = -1 * q_max
        return q_min, q_max

    def scale2minmax(scale, zerop, is_signed, qbits, symmetric):
        q_min, q_max = get_qmin_qmax(is_signed, qbits)

        if not symmetric:
            fmin = (zerop+q_min)/scale
            fmax = (q_max-q_min)/scale+fmin
        else:
            fmax = (q_max-q_min)/scale/2
            fmin = -fmax
        return fmin, fmax

    def get_denoise_max_min(out):
        have_inf = out.betensor.float() < -32768
        mvalue = have_inf*-127
        filt_out = out.betensor.float()*(~have_inf)+mvalue
        U, S, Vh = torch.linalg.svd(filt_out, full_matrices=False)
        percent = thresh
        S_sorted = S.flatten().sort()[0]
        slen = S_sorted.shape[0]
        the = S_sorted[int(slen*percent)]
        while the > 1 and percent > 0.01:
            the = S_sorted[int(slen*percent)]
            percent = percent - 0.004
        if the > 1.56:
            the = 1.56
        while the < 0.0001 and percent < 0.2:
            the = S_sorted[int(slen*percent)]
            percent = percent + 0.015
        if the < 0.0001:
            the = 0.0018
        S1 = S*(torch.abs(S) > the)
        if (S1 == torch.tensor(0)).all():
            bmin = out.betensor.min()
            bmax = out.betensor.max()
        else:
            denoise_t = U @ torch.diag_embed(S1) @ Vh
            bmin = denoise_t.min()
            bmax = denoise_t.max()
        if out.betensor.min() > 0:
            bmin = torch.tensor(0)
            bmax = bmax-bmin
        return bmin.item(), bmax.item()

    def check_node_if_optimization(k, nodes, oplist):
        layer_n = nodes[k]
        mscopes = oplist
        if len(mscopes) < 1:
            return False
        else:
            return layer_n.type.name.lower() in mscopes or int(layer_n.attrs['layer_id']) in mscopes

    def get_best_scale(out, bmin, bmax, quant_mode, op_list):

        out_signed = is_signed(out.dtype)
        q_min, q_max = get_qmin_qmax(out_signed, out.qbits)
        have_inf = out.betensor.float() < -32768
        mvalue = have_inf*-127
        filt_out = out.betensor.float()*(~have_inf)+mvalue
        U, S, Vh = torch.linalg.svd(filt_out, full_matrices=False)
        denoise_out = PyTensor('denoise_out', out.ir_shape, out.dtype)
        denoise_out.min, denoise_out.max = bmin, bmax

        scale, zerop, _, _, _ = get_linear_quant_params_from_tensor(denoise_out, quant_mode, out.qbits, out_signed)

        best_s = scale
        # if str(n.type)[7:].lower() in op_list:
        zero = zerop
        if op_list:

            init_s = scale*scale
            qbetensor = linear_quantize_clip(filt_out, init_s, zerop, q_min, q_max)
            deqbetenor = linear_dequantize(qbetensor, init_s, zerop)
            qU, qS, qVh = torch.linalg.svd(deqbetenor, full_matrices=False)
            min_dist = torch.dist(S, qS)
            # zero = torch.clamp(zerop.round(), -2 ** (out.qbits - 1) + 1, 2 ** (out.qbits - 1))
            # TODO: support per-channel
            if is_torch_tensor_with_multi_data(scale):
                OPT_ERROR(f"svd now only support per-tensor quantization.")
            start, end = alpha * scale.item(), beta * scale.item()
            for s in torch.linspace(start, end, nsteps):
                qbetensor = linear_quantize_clip(filt_out, s, zerop, q_min, q_max)
                deqbetenor = linear_dequantize(qbetensor, s, zerop)
                qU, qS, qVh = torch.linalg.svd(deqbetenor, full_matrices=False)
                dist = torch.dist(S, qS)
                if dist < min_dist:
                    min_dist = dist
                    best_s = s

        return best_s, zero

    def filter(input, init_state, alpha):
        yn_1 = init_state
        yn_2 = 0
        yn_3 = 0
        alpha_y_1 = alpha*1.53/3
        alpha_y_2 = alpha*1.44/3
        alpha_y_3 = alpha*0.03/3
        alpha_x = 1-alpha_y_1-alpha_y_2-alpha_y_3
        for i in range(len(input)):
            yout = alpha_y_1*yn_1+alpha_y_2*yn_2+alpha_y_3*yn_3+alpha_x*input[i]
            yn_3 = yn_2
            yn_2 = yn_1
            yn_1 = yout
        return yout.item()

    # prevent deleting intermediate tensors
    g.ref_count_tensors = {}

    vdataloader = copy.deepcopy(cdataloader)
    dataset_len = len(vdataloader.dataset.data)
    start = 0
    end = dataset_len
    with tqdm(total=dataset_len, desc='svd_based_search_scale', file=sys.stdout, leave=True) as pbar:
        # need adjust min max by denoise
        # calibraton_min_max_op = ['eltwise','pooling','convolution']
        # need adjust scale according calibrated min max
        # calibration_scale_op = ['mul']
        # do nothing
        # bypass_op = []
        # group_norm_op = [OpType.LayerNorm,OpType.InstanceNorm,OpType.GroupNorm]
        tscale = torch.ones(dataset_len, len(g.nodes), 2)*-1
        need_adjust_nodes = []
        # prevent deleting intermediate tensors
        g.ref_count_tensors = {}
        for i, sample in enumerate(vdataloader):
            if i >= start and i < end:
                inp_data, _ = sample
                need_adjust_nodes.clear()
                g.feed_inputs_data(inp_data)

                for k, n in enumerate(g.nodes):
                    n.forward()
                    # if n.outputs[0].betensor.ndim<=1 or str(n.type)[7:].lower() in oplist[2]:
                    # mode 2 is only denoise op which not in oplist[2]
                    # mode 1 denoise and search scale then update fmin, fmax
                    # mode 0 assume min max,search scale again
                    optimized_flag = check_node_if_optimization(k, g.nodes, oplist[int(mode)])
                    if n.outputs[0].betensor.ndim <= 1:
                        continue
                    statistic_momentum = n.attrs["running_statistic_momentum"]
                    if i == start:
                        statistic_momentum = 0
                    q_bits_activation = n.attrs["q_bits_activation"]
                    quant_mode = n.attrs["q_mode_activation"]

                    out = n.outputs[0]

                    if optimized_flag and mode > 0:
                        fmin, fmax = get_denoise_max_min(out)
                        if mode == 1:
                            best_s, zerop = get_best_scale(out, fmin, fmax, quant_mode, optimized_flag)
                            symmetric = QuantMode.is_symmetric(quant_mode)

                            out_signed = is_signed(n.outputs[0].dtype)
                            fmin, fmax = scale2minmax(best_s, zerop, out_signed, q_bits_activation, symmetric)
                        tscale[i, k, 0] = fmin
                        tscale[i, k, 1] = fmax
                        need_adjust_nodes.append(k)
            pbar.update(1)

        for i in range(len(need_adjust_nodes)):
            k = need_adjust_nodes[i]
            tfmin = tscale[start:end, k, 0]
            tfmax = tscale[start:end, k, 1]
            n = g.nodes[k]
            statistic_momentum = n.attrs["running_statistic_momentum"]
            quant_mode = n.attrs["q_mode_activation"]
            fmax = filter(tfmax, tfmax[0], statistic_momentum)
            fmin = filter(tfmin, tfmin[0], statistic_momentum)

            # if n.type not in group_norm_op:
            #     best_s, zerop = get_best_scale(n.outputs[0], fmin, fmax, quant_mode,
            #                                    check_node_if_optimization(k, g.nodes, oplist[0]))
            #     out_signed = is_signed(n.outputs[0].dtype)
            #     fmin,fmax = scale2minmax(best_s, zerop,out_signed, q_bits_activation,symmetric)
            n.outputs[0].max = fmax
            n.outputs[0].min = fmin
            # if n.type in group_norm_op:
            #     for j in range(len(n.placeholders)):
            #         out = n.placeholders[j]
            #         fmin, fmax = out.min, out.max
            #         best_s, zerop = get_best_scale(out, fmin, fmax, quant_mode,
            #                                        check_node_if_optimization(k, g.nodes, oplist[0]))
            #         out_signed = is_signed(out.dtype)
            #         out.min,out.max = scale2minmax(best_s, zerop,out_signed, q_bits_activation,False)

        for k, n in enumerate(g.nodes):
            # n.forward()
            if n.outputs[0].betensor.ndim <= 1:
                continue
            if check_node_if_optimization(k, g.nodes, oplist[0]):
                quant_mode = n.attrs["q_mode_activation"]
                q_bits_activation = n.attrs["q_bits_activation"]
                symmetric = QuantMode.is_symmetric(quant_mode)
                fmax = n.outputs[0].max
                fmin = n.outputs[0].min
                best_s, zerop = get_best_scale(n.outputs[0], fmin, fmax, quant_mode, True)
                out_signed = is_signed(n.outputs[0].dtype)
                fmin, fmax = scale2minmax(best_s, zerop, out_signed, q_bits_activation, symmetric)
                n.outputs[0].max = fmax
                n.outputs[0].min = fmin

        pbar.refresh()
