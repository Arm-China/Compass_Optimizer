# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

import torch


def nkld_calibration(t, *args):  # eg. 5kld/10kld/20kld
    cstrategy = args[0]
    tn = 10 if len(cstrategy) < 4 else max(2, int(cstrategy[:-3]))
    t.min, t.max = tune_min_max_by_kld_v1(t.running_histc, t.running_min, t.running_max, 2 ** t.qbits, tn)
    if None != t.running_histc_key_axis:
        sk_min = []
        sk_max = []
        for k, histc in enumerate(t.running_histc_key_axis):
            smin, smax = tune_min_max_by_kld_v1(histc,
                                                t.running_min_key_axis[k],
                                                t.running_max_key_axis[k],
                                                2 ** t.qbits,
                                                tn)
            sk_min.append(smin)
            sk_max.append(smax)
        t.min_key_axis = torch.tensor(sk_min,
                                      dtype=t.running_max_key_axis.dtype,
                                      device=t.running_max_key_axis.device)
        t.max_key_axis = torch.tensor(sk_max,
                                      dtype=t.running_max_key_axis.dtype,
                                      device=t.running_max_key_axis.device)
    elif None != t.running_min_key_axis:
        t.min_key_axis = t.running_min_key_axis
        t.max_key_axis = t.running_max_key_axis


def tune_min_max_by_kld_v1(histc, min_v, max_v, target_bins, topk=10):
    hbins = histc.numel()
    if hbins <= target_bins or target_bins < 2:
        return min_v, max_v
    interval = (max_v - min_v) / hbins
    midx = torch.argmax(histc).item()  # align on the max element's position
    lidx = 0
    ridx = hbins-1
    if min_v >= 0:
        lidx = 0
        ridx = target_bins-1
    elif max_v <= 0:
        lidx = hbins - target_bins
        ridx = hbins-1
    else:
        if midx < hbins - 1 - midx:
            lidx = max(midx - (target_bins // 2), 0)
            ridx = lidx + target_bins - 1
        else:
            ridx = min(midx + (target_bins // 2), hbins-1)
            lidx = ridx - (target_bins - 1)
    steps = max(lidx+1, hbins-ridx)

    p = torch.nn.functional.pad(histc, (steps-(lidx+1), steps-(hbins-ridx)))
    bins = p.numel()
    h_cumsum = p.cumsum(dim=0, dtype=torch.float32)
    h_cumsum_r = p.flip(0).cumsum(dim=0, dtype=torch.float32).flip(0)
    p = p.repeat(steps, 1)
    triu = torch.ones(steps, steps, device=p.device).triu()
    mask = torch.nn.functional.pad(triu.flip(1), (0, bins-steps)) + torch.nn.functional.pad(triu, (bins-steps, 0))
    p = torch.where(mask > 0, torch.zeros_like(p), p)
    # add outliers to border bins
    triu = torch.diag_embed(torch.ones(steps, device=p.device))
    border = torch.nn.functional.pad(triu.flip(1), (0, bins-steps))
    p = torch.where(border > 0, h_cumsum, p)
    border = torch.nn.functional.pad(triu, (bins-steps, 0))
    p = torch.where(border > 0, h_cumsum_r, p)
    # normalize p
    p = p / h_cumsum[-1]

    # get extended qbins matrix steps * (target_bins*2) X max_qinterval
    qbins_seq = []
    max_qinterval = bins // target_bins
    qindex = torch.zeros_like(p, dtype=torch.long)
    ts = torch.arange(target_bins, device=p.device).reshape(-1, 1)
    for idx in range(steps-1, -1, -1):
        cur_p_len = bins - 2*idx
        cur_p = p.select(0, steps-1-idx).narrow(0, idx, cur_p_len)
        qinterval, qtail = divmod(cur_p_len, target_bins)
        cur_p = torch.nn.functional.pad(cur_p, (0, qinterval * target_bins - qtail))
        qbins = cur_p.reshape(target_bins * 2, qinterval)
        qbins = torch.nn.functional.pad(qbins, (0, max_qinterval - qinterval))
        qbins_seq.append(qbins)
        cur_index = qindex.narrow(0, steps-1-idx, 1).narrow(1, idx, cur_p_len)
        cur_ts = torch.nn.functional.pad(ts.repeat(1, qinterval).reshape(1, -1), (0, qtail), value=target_bins-1)
        cur_index.index_copy_(0, torch.tensor([0], device=p.device), cur_ts)
    qbins_matrix = torch.cat(qbins_seq, 0)
    qbins_sum = qbins_matrix.sum(dim=1)
    qbins_nonz = qbins_matrix.count_nonzero(dim=1)
    # merge [target_bins, 2*target_bins) values into target_bins, and get steps X target_bins qbins_sum & qbins_nonz
    qbins_sum = qbins_sum.reshape(steps, target_bins*2)
    qbins_sum.narrow(1, 0, target_bins).index_add_(1, torch.tensor([target_bins-1], device=p.device),
                                                   qbins_sum.narrow(1, target_bins, target_bins).sum(dim=1).reshape(-1, 1))
    qbins_sum = qbins_sum.narrow(1, 0, target_bins)
    qbins_nonz = qbins_nonz.reshape(steps, target_bins*2)
    qbins_nonz.narrow(1, 0, target_bins).index_add_(1, torch.tensor([target_bins-1], device=p.device),
                                                    qbins_nonz.narrow(1, target_bins, target_bins).sum(dim=1).reshape(-1, 1))
    qbins_nonz = qbins_nonz.narrow(1, 0, target_bins)
    eps = torch.finfo(torch.float32).eps
    qvalues = qbins_sum / (qbins_nonz + eps)
    # get q
    qtmp = torch.gather(qvalues, 1, qindex)
    p_eps = p + eps
    c_q = torch.where(p != 0.0, qtmp, p_eps)
    c_p = torch.where(p != 0.0, p, p_eps)
    # get kl divergences
    kl_divergence = (p * torch.log(c_p / c_q)).sum(dim=1)

    min_idx = torch.argsort(kl_divergence)[:topk].max()  # make it more stable
    lval = min_v + max(lidx-min_idx, 0) * interval
    rval = max_v - (hbins-1 - min(ridx+min_idx, hbins-1)) * interval
    if isinstance(lval, torch.Tensor):
        lval = lval.item()
    if isinstance(rval, torch.Tensor):
        rval = rval.item()
    return min(lval, 0.0), max(rval, 0.0)
