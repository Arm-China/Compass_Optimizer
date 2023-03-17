# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils.quant_tool_utils import cosine_distance
from AIPUBuilder.Optimizer.logger import *
import torch
import torch.fft
import re


def weighted_scale_param_calibration(t, *args):
    cstrategy = args[0]
    opt_v_list = re.findall('[0-9]+\.?[0-9]*\s*', cstrategy.lower())
    param = [float(opt_i) for opt_i in opt_v_list]

    t.min, t.max = get_min_max_params_from_tensor(t, t.qbits, param, t.extrema_min < 0.0)
    if None != t.running_min_key_axis:
        t.min_key_axis = t.running_min_key_axis
        t.max_key_axis = t.running_max_key_axis


class static_varibale:
    xstd = 0
    xnorm_std = 0
    xnorm_std_postive = 0
    cnt = 0
    optimize_param = [0.685, 0.875, 0.7175, 0.01965]
    alpha = [0.975, 0.910, 0.9956]


def calc_sid(x, y, eps):

    sum_x = torch.sum(x)+eps
    sum_y = torch.sum(y)+eps
    p_i = x/sum_x+eps
    q_i = y/sum_y+eps
    sid = (torch.sum(p_i*torch.log(p_i/q_i)) + torch.sum(q_i*torch.log(q_i/p_i))).item()
    return sid


def get_min_max_params_from_tensor(x, bits, weight_scale_optimize, is_signed):
    QUANTIZE_ZERO_BAND = torch.finfo(torch.float32).eps
    auto_calc_alpha = weight_scale_optimize[0] > 1

    if auto_calc_alpha:
        static_varibale.cnt += 1
        dim_id = 1
        if x.betensor.dim() == 1:
            dim_id = 0
        sample_num = 7
        xnorm_std = torch.std(torch.nn.functional.normalize(x.betensor.float(), dim=dim_id))
        xnorm_std_postive = torch.std((torch.nn.functional.normalize(
            x.betensor.float(), dim=dim_id) > 0)*torch.nn.functional.normalize(x.betensor.float(), dim=dim_id))
        xnorm_std = torch.where(torch.isnan(xnorm_std), torch.ones_like(xnorm_std)*QUANTIZE_ZERO_BAND, xnorm_std).item()
        xnorm_std_postive = torch.where(torch.isnan(xnorm_std_postive), torch.ones_like(
            xnorm_std_postive)*QUANTIZE_ZERO_BAND, xnorm_std_postive).item()
        if static_varibale.cnt < sample_num:
            static_varibale.xnorm_std += xnorm_std
            static_varibale.xnorm_std_postive += xnorm_std_postive
        elif static_varibale.cnt % 2 == 0 or static_varibale.cnt < 300:
            xstd_thr = static_varibale.xnorm_std/(sample_num-1)
            xstd_postive_thr = static_varibale.xnorm_std_postive/(sample_num-1)
            if xnorm_std > xstd_thr:
                static_varibale.optimize_param[0] = (1-static_varibale.alpha[0]) * \
                    xnorm_std+static_varibale.alpha[0]*static_varibale.optimize_param[0]
                if static_varibale.optimize_param[0] > 0.8285 or static_varibale.optimize_param[0] < 0.65:
                    static_varibale.alpha[0] = 0.9998
            elif xnorm_std_postive < xstd_postive_thr and xnorm_std < xstd_thr*0.75:
                static_varibale.optimize_param[2] = (1-static_varibale.alpha[2]) * \
                    xnorm_std+static_varibale.alpha[2]*static_varibale.optimize_param[2]
                if static_varibale.optimize_param[2] < 0.7012:
                    static_varibale.alpha[2] = 0.99998
            else:
                static_varibale.optimize_param[1] = (1-static_varibale.alpha[1]) * \
                    xnorm_std+static_varibale.alpha[1]*static_varibale.optimize_param[1]
                if static_varibale.optimize_param[1] < 0.475:
                    static_varibale.alpha[1] = 0.9998

        static_varibale.xstd += x.running_std if x.running_std < 1 else 0
        std_mean = static_varibale.xstd/static_varibale.cnt
        static_varibale.optimize_param[3] = (1-static_varibale.alpha[2])*std_mean + \
            static_varibale.alpha[2]*static_varibale.optimize_param[3]
        rparam = static_varibale.optimize_param
        OPT_DEBUG("recomend weighted_scale_param for tensor '%s' is:\n %.4f, %.4f, %.4f, %.4f " %
                  (x.name, rparam[0], rparam[1], rparam[2], rparam[3]), end='\r')

    q_max, q_min = 2 ** bits - 1, 0
    if is_signed:
        q_max = 2 ** (bits - 1) - 1
        q_min = -1 * q_max-1
    q_range = q_max - q_min

    f_ranges_pos = torch.tensor(abs(max(0.0, x.running_max)), dtype=torch.float32, device=x.betensor.device)
    f_ranges_neg = torch.tensor(abs(min(0.0, x.running_min)), dtype=torch.float32, device=x.betensor.device)
    f_ranges = [f_ranges_pos, f_ranges_neg]

    q_ranges = q_range * torch.ones_like(f_ranges_pos)
    f_ranges[0] = torch.where(f_ranges[0] < QUANTIZE_ZERO_BAND, q_ranges, f_ranges[0])
    f_ranges[1] = torch.where(f_ranges[1] < QUANTIZE_ZERO_BAND, q_ranges, f_ranges[1])
    scale_all = q_ranges / f_ranges[0], q_ranges / f_ranges[1]

    if auto_calc_alpha:
        x_v = torch.flatten(x.betensor)

        eps = torch.finfo(torch.float32).eps
        zero_cnt = ((torch.abs(x_v) < eps)*1).sum()
        sort_xv = q_ranges/torch.sort(torch.abs(x_v), -1, True)[0]  # top zero_cnt
        continue_cnt = 0
        most_cnt = 0
        pre_q_zero_cnt = x_v.shape[0]

        X_V = torch.abs(torch.fft.rfft(x_v))
        # init initial similarity
        gama = 0.97
        beta = 2
        delta = zero_cnt/x_v.shape[0]
        if f_ranges_pos < f_ranges_neg:
            alpha = 0.0001+delta
            stepsize = 0.0009
        else:
            alpha = 1-0.0001-delta
            stepsize = -0.0009
        k = 4
        if sort_xv.shape[0] < 6:
            k = sort_xv.shape[0]//2
        best_scale = alpha*scale_all[0]+(1-alpha-delta)*scale_all[1]+delta * \
            torch.tensor([sort_xv[k], scale_all[0], scale_all[1]]).max()

        if zero_cnt > 1:
            q_x_v = torch.clamp(torch.round(best_scale * x_v), q_min, q_max)
            q_zero_cnt = ((torch.abs(q_x_v) < eps)*1).sum()
            if q_zero_cnt/x_v.shape[0] > 0.008:
                gama = 0
                beta = 0.96
                k = 5
        most_similar = [1, 0]

        while 1-alpha-delta > 0 and alpha > 0:
            scale = alpha*scale_all[0]+(1-alpha-delta)*scale_all[1] + delta * \
                torch.tensor([sort_xv[k], scale_all[0], scale_all[1]]).max()
            q_x_v = torch.clamp(torch.round(scale * x_v), q_min, q_max)
            q_zero_cnt = ((torch.abs(q_x_v) < eps)*1).sum()

            q_X_V = torch.abs(torch.fft.rfft(q_x_v))
            sid_sim = calc_sid(X_V[:], q_X_V[:], eps)
            sim = cosine_distance(x_v[:], q_x_v[:])
            if (sid_sim < most_similar[0]*gama or sim > most_similar[1]*beta):
                continue_cnt = continue_cnt + 1

            elif continue_cnt > 0 or q_zero_cnt > pre_q_zero_cnt:
                continue_cnt = 0
            if continue_cnt > 2 and continue_cnt > most_cnt*beta:
                best_scale = scale
                most_cnt = continue_cnt
                most_similar = [sid_sim, sim]
            alpha = alpha + stepsize
            pre_q_zero_cnt = q_zero_cnt
        scale = best_scale
    else:
        if f_ranges_pos > f_ranges_neg:
            alpha = weight_scale_optimize[0]  # 0.7263,0.0875,0.365 acc 0.435989 ,std #0.01995
        else:
            if x.running_std > weight_scale_optimize[3]:  # 0.01995
                alpha = weight_scale_optimize[1]
            else:
                alpha = weight_scale_optimize[2]
        scale = alpha*scale_all[0]+(1-alpha)*scale_all[1]

    if is_signed:
        fmax = q_ranges/scale
        fmin = -fmax
    else:
        fmax = q_ranges/scale
        fmin = fmax - fmax
    if fmax.dim() < 1:
        return fmin.item(), fmax.item()
    else:
        return fmin, fmax
