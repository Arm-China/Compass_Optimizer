# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import tqdm, OPT_ERROR
import torch
import sys
from torch.nn import MSELoss as mseloss
import torch.nn as nn
from AIPUBuilder.Optimizer.ops.activation import with_activation_out_is_signed


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
    from AIPUBuilder.Optimizer.config import PerNodeFieldDict
    dpdict = PerNodeFieldDict(False)
    oplist = (dpdict, dpdict, dpdict, dpdict)
    if int(mode) <= 0:
        oplist = (mscopes, dpdict, dpdict, dpdict)
    elif int(mode) == 1:
        oplist = (dpdict, mscopes, dpdict, dpdict)
    elif int(mode) == 2:
        oplist = (dpdict, dpdict, mscopes, dpdict)
    elif int(mode) >= 3:
        oplist = (dpdict, dpdict, dpdict, mscopes)
    _svd_based_search_scale(g, cdataloader, alpha, beta, nsteps, thresh, mode, oplist)


def _svd_based_search_scale(g, cdataloader, alpha, beta, nsteps, thresh, mode, oplist):
    import copy

    def get_qmin_qmax(outsign, qbits, dtype):
        if is_float(dtype):
            q_min, q_max = dtype2range(dtype)
        else:
            q_max, q_min = 2 ** qbits - 1, 0
            if outsign:
                if QuantMode.is_full_range(quant_mode):
                    q_max = 2 ** (qbits - 1) - 1
                    q_min = -1 * q_max - 1
                else:
                    q_max = 2 ** (qbits - 1) - 1
                    q_min = -1 * q_max
        return q_min, q_max

    # def scale2minmax(scale, zerop, is_signed, qbits, symmetric):
    #     q_min, q_max = get_qmin_qmax(is_signed, qbits)

    #     if not symmetric:
    #         fmin = (zerop+q_min)/scale
    #         fmax = (q_max-q_min)/scale+fmin
    #     else:
    #         fmax = (q_max-q_min)/scale/2
    #         fmin = -fmax
    #     return fmin, fmax

    def get_denoise_max_min(out):
        # have_inf = (out.betensor.float() < -32768)
        # mvalue = have_inf*-127
        # filt_out = out.betensor.float()*(~have_inf)+mvalue
        quant_type = out.attrs.get('quant_type')
        filt_out = torch.clamp(out.betensor.float(), min=-32767)
        U, S, Vh = torch.linalg.svd(filt_out, full_matrices=False)
        percent = thresh
        S_sorted = S.flatten().sort()[0]
        slen = S_sorted.shape[0]
        the = S_sorted[int(slen*percent)]
        while the > 1 and percent > 0.01:
            the = S_sorted[int(slen*percent)]
            percent = percent - 0.002
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
        return mscopes.get(layer_n)

    def get_best_scale(n, bmin, bmax, op_list):
        out = n.outputs[0]
        quant_mode = n.attrs["q_mode_activation"]
        quant_type = n.attrs.get('quant_type')
        if quant_type in ['fp8_e4m3fn', 'fp8_e5m2']:
            q_type_activation = QuantType._to_Dtype(QuantType.activation_type(quant_type))
        else:
            q_type_activation = out.dtype
        is_float_quant = is_float(q_type_activation)
        out_signed = is_signed(out.dtype)
        q_min, q_max = get_qmin_qmax(out_signed, out.qbits, q_type_activation)
        have_inf = out.betensor.float() < -32768
        mvalue = have_inf*-127
        filt_out = out.betensor.float()*(~have_inf)+mvalue
        # U, S, Vh = torch.linalg.svd(filt_out, full_matrices=False)
        S = torch.abs(torch.fft.fft(filt_out, norm='forward'))
        denoise_out = PyTensor('denoise_out', out.ir_shape, q_type_activation)
        denoise_out.min, denoise_out.max = bmin, bmax

        if is_float_quant:
            scale, zerop, _, _ = get_fpx_quant_params_from_tensor(denoise_out, quant_mode, q_type_activation)
        else:
            scale, zerop, _, _, _ = get_linear_quant_params_from_tensor(
                denoise_out, quant_mode, out.qbits, out_signed)

        best_s = 1
        scale = scale.item()
        # if str(n.type)[7:].lower() in op_list:
        zero = zerop
        if op_list:

            init_s = scale
            # qbetensor = linear_quantize_clip(
            #     filt_out, init_s, zerop, q_min, q_max)
            qbetensor = linear_quantize_clip(filt_out.float(), init_s, zerop,
                                             q_min, q_max, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', q_type_activation))

            deqbetenor = linear_dequantize(qbetensor, init_s, zerop)
            # qU, qS, qVh = torch.linalg.svd(deqbetenor, full_matrices=False)
            qS = torch.abs(torch.fft.fft(deqbetenor, norm='forward'))
            min_dist = MSE(S, qS)
            # zero = torch.clamp(zerop.round(), -2 ** (out.qbits - 1) + 1, 2 ** (out.qbits - 1))
            # TODO: support per-channel
            if is_torch_tensor_with_multi_data(scale):
                OPT_ERROR(f"svd now only support per-tensor quantization.")
            start, end = alpha, beta
            for f in torch.linspace(start, end, nsteps):
                denoise_out.min, denoise_out.max = bmin*f, bmax*f
                if is_float_quant:
                    s, zerop, _, _ = get_fpx_quant_params_from_tensor(denoise_out, quant_mode, q_type_activation)
                else:
                    s, zerop, _, _, _ = get_linear_quant_params_from_tensor(
                        denoise_out, quant_mode, out.qbits, out_signed)

                # qbetensor = linear_quantize_clip(
                #     filt_out, s, zerop, q_min, q_max)
                qbetensor = linear_quantize_clip(filt_out.float(), s, zerop,
                                                 q_min, q_max, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', q_type_activation))
                deqbetenor = linear_dequantize(qbetensor, s, zerop)
                qS = torch.abs(torch.fft.fft(deqbetenor, norm='forward'))
                # qU, qS, qVh = torch.linalg.svd(deqbetenor, full_matrices=False)
                # dist = torch.dist(S, qS)
                dist = MSE(S, qS)
                if dist < min_dist:
                    min_dist = dist
                    best_s = f

        return bmin*best_s, bmax*best_s

    def linear_op_quantize_param_search(n, alpha, beta, nsteps, thresh):
        quant_type = n.attrs.get('quant_type')
        q_mode_bias = n.attrs["q_mode_bias"]
        q_mode_weight = n.attrs["q_mode_weight"]
        q_mode_activation = n.attrs["q_mode_activation"]
        if q_mode_weight != q_mode_bias:
            OPT_FATAL("Currently quantization mode of weight (q_mode_weight) and bias (q_mode_bias) must be the same!")
        q_bits_weight = n.attrs["q_bits_weight"]
        q_bits_bias = n.attrs["q_bits_bias"]
        q_bits_activation = n.attrs["q_bits_activation"]
        multiplier_bits = n.get_attrs('multiplier_bits', optional=True, default_value=q_bits_activation)

        inp = n.inputs[0]

        w = n.constants["weights"]

        key_axis = w.key_axis

        out = n.outputs[0]
        out_signed = with_activation_out_is_signed(n) or n.force_dtype_int
        out.qbits = q_bits_activation

        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, q_bits_activation, is_signed=out_signed)
        out.qinvariant = False

        inp_scale = inp.scale

        # simulate result

        MSE = mseloss()
        best_s = 1
        w.qbits = q_bits_weight
        w.qinvariant = False

        w_sim = PyTensor('w_sim', w.ir_shape, w.ir_dtype)
        w_sim = w.clone(w_sim)

        worst_cosim = 1
        best_mse = 1000
        np.random.seed(10000+int(n.attrs['layer_id']))

        # # for s in torch.linspace(w.broadcast_scale.item()*0.65, w.broadcast_scale.item()*1.65, 100) :
        # init_cnt = 0
        # fftsize = 2048
        for f in torch.linspace(alpha, beta, nsteps):
            # btensor = torch.tensor(np.random.randint(inp.qmin*thresh,inp.qmax*thresh,inp.ir_shape),device=inp.device)
            uniform = np.random.uniform(-1, 1, inp.ir_shape)*inp.qmax
            itensor = torch.tensor(uniform, device=inp.device)
            freq_tensor = torch.fft.rfft(itensor)
            spectrum = torch.abs(freq_tensor)
            # w_spectrum = torch.abs(torch.fft.rfft(w.betensor))
            half_spec = (spectrum < spectrum.max(dim=0)[
                         0]*0.5)*thresh + (spectrum >= spectrum.max(0)[0]*0.5)*1.15
            fft_btensor = freq_tensor*half_spec
            btensor = torch.clamp(torch.fft.irfft(
                fft_btensor), inp.qmin, inp.qmax).int()
            # btensor = torch.tensor(np.random.uniform(-1, 1, inp.ir_shape)*inp.qmax*thresh,device=inp.device).int()
            if QuantMode.is_per_channel(q_mode_weight):
                w_sim.min_key_axis, w_sim.max_key_axis = w.min_key_axis*f, w.max_key_axis*f
            else:
                w_sim.min, w_sim.max = w.min*f, w.max*f
            w_sim.scale, w_sim.zerop, w_sim.qmin, w_sim.qmax, w_sim.dtype = get_linear_quant_params_from_tensor(
                w_sim, q_mode_weight, q_bits_weight, is_signed=True)
            # btensor = torch.tensor(np.random.randint(w.qmin,w.qmax,inp.ir_shape),device=inp.device)
            # fbetensor = torch.tensor(np.random.rand(w.qmin,w.qmax,inp.ir_shape),device=inp.device)
            weights = linear_quantize_clip(
                w.betensor, w_sim.broadcast_scale,  w_sim.broadcast_zerop, w.qmin, w.qmax)
            if 'biases' in n.constants:
                b = n.constants["biases"]
                b.scale = inp_scale * w_sim.scale
                b.zerop = 0
                b.qmin = -2 ** (q_bits_bias - 1)
                b.qmax = 2 ** (q_bits_bias - 1) - 1
                b.qbits = q_bits_bias
                bias = linear_quantize_clip(
                    b.betensor, b.broadcast_scale, 0, b.qmin, b.qmax)

                b.dtype = bits2dtype(b.qbits, is_signed=True)
                b.qinvariant = False
            local_rescale = out.scale / (inp_scale * w_sim.scale)
            do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(local_rescale,
                                                                                              mult_bits=multiplier_bits,
                                                                                              force_shift_positive=n.force_shift_positive)

            x = nn.functional.linear(btensor.float(), weights, bias,)

            x = linear_requantize(x, do_scale, do_shift, n.outputs[0].broadcast_zerop,
                                  n.outputs[0].qmin, n.outputs[0].qmax, out.key_axis)

            fout = linear_dequantize(x, out.scale, out.zerop, key_axis=None)

            fbtensor = linear_dequantize(
                btensor, inp_scale, n.inputs[0].broadcast_zerop, key_axis=None)
            fx = nn.functional.linear(
                fbtensor, w.betensor.float(), b.betensor.float(),)

            qS = torch.abs(torch.fft.fft(weights))
            S = torch.abs(torch.fft.fft(w.betensor.float()))

            # qw = torch.fft.fft(weights)
            # wfft = torch.fft.fft(w.betensor.float())
            # wqwf  =qw*wfft
            # cosim = torch.sum(abs(wqwf))/torch.norm(abs(qw))/torch.norm(abs(wfft))
            cosim = cosine_distance(qS, S)
            if cosim < worst_cosim:
                worst_cosim = cosim

            # plt.plot(qS.flatten().cpu().numpy())
            # plt.plot(S.flatten().cpu().numpy())
            # plt.show()

            # qS = torch.abs(torch.fft.fft(fout))
            # S = torch.abs(torch.fft.fft(fx))
            # U, S, Vh = torch.linalg.svd(fout, full_matrices=False)
            # qU, qS, qVh = torch.linalg.svd(fx, full_matrices=False)
            # sim =cosine_distance(qS,S)
            mse = MSE(fx, fout).item()

            # mse = MSE(qS,S).item()
            # cosim = cosine_distance(qS,S)
            if mse < best_mse*worst_cosim:
                best_mse = mse*worst_cosim + (1-worst_cosim)*best_mse
                best_s = f

        if QuantMode.is_per_channel(q_mode_weight):
            w.min_key_axis = w.min_key_axis*best_s
            w.max_key_axis = w.max_key_axis*best_s
        else:
            w.min = w.min*best_s
            w.max = w.max*best_s

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
    dataset_len = len(vdataloader.dataset)
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
        MSE = mseloss()
        # cos_sim = nn.CosineSimilarity(eps=1e-8)
        # prevent deleting intermediate tensors
        g.ref_count_tensors = {}
        if oplist[3].rdict or oplist[3].tdict:
            for k, n in enumerate(g.nodes):
                if n.attrs["q_bits_activation"] <= 8 and n.type == OpType.FullyConnected and check_node_if_optimization(k, g.nodes, oplist[3]):
                    linear_op_quantize_param_search(
                        n, alpha, beta, nsteps, thresh)
        else:
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
                        optimized_flag = check_node_if_optimization(
                            k, g.nodes, oplist[int(mode)])
                        if n.outputs[0].betensor.ndim <= 1:
                            continue
                        statistic_momentum = n.attrs["running_statistic_momentum"]
                        if i == start:
                            statistic_momentum = 0
                        q_bits_activation = n.attrs["q_bits_activation"]
                        quant_mode = n.attrs["q_mode_activation"]

                        out = n.outputs[0]

                        if optimized_flag and mode > 0 and mode < 3:
                            omin, omax = get_denoise_max_min(out)
                            if mode == 1:
                                omin, omax = get_best_scale(
                                    n, omin, omax, optimized_flag)
                                # symmetric = QuantMode.is_symmetric(quant_mode)

                                # out_signed = is_signed(n.outputs[0].dtype)
                                # fmin,fmax = scale2minmax(best_s, zerop,out_signed, q_bits_activation,symmetric)
                            tscale[i, k, 0] = omin
                            tscale[i, k, 1] = omax
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
            if oplist[0].tdict:
                for k, n in enumerate(g.nodes):
                    # n.forward()
                    if n.outputs[0].betensor.ndim <= 1:
                        continue
                    if check_node_if_optimization(k, g.nodes, oplist[0]):
                        # quant_mode = n.attrs["q_mode_activation"]
                        q_bits_activation = n.attrs["q_bits_activation"]
                        symmetric = QuantMode.is_symmetric(quant_mode)
                        fmax = n.outputs[0].max
                        fmin = n.outputs[0].min
                        omin, omax = get_best_scale(
                            n, fmin, fmax, True)
                        # out_signed = is_signed(n.outputs[0].dtype)
                        # fmin,fmax = scale2minmax(best_s, zerop,out_signed, q_bits_activation,symmetric)
                        n.outputs[0].max = omax
                        n.outputs[0].min = omin

        pbar.refresh()
