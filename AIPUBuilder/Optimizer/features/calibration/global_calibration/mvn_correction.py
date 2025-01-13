# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
from AIPUBuilder.Optimizer.logger import OPT_INFO
import torch
import sys


def mvn_correction_global_calibration(g, cdataloader, mparams, mscopes):
    vec = mparams
    mode = int(vec[0] if len(vec) > 0 else 0)
    alpha = float(vec[1] if len(vec) > 1 else (0.5 if mode <= 0 else 1.0))
    beta = int(vec[2] if len(vec) > 2 else 0)
    gamma = int(vec[3] if len(vec) > 3 else 0)
    act_bits = int(vec[4] if len(vec) > 4 else 16)
    wgt_bits = int(vec[5] if len(vec) > 5 else 16)
    bias_bits = int(vec[6] if len(vec) > 6 else 48)
    lut_bits = int(vec[7] if len(vec) > 7 else 10)
    multi_bits = int(vec[8]) if len(vec) > 8 else act_bits

    msg = (f"mvn_correction with mode={mode}, alpha={alpha}, beta={beta}, gamma={gamma}, act_bits={act_bits}, "
           f"wgt_bits={wgt_bits}, bias_bits={bias_bits}, lut_bits={lut_bits}, multi_bits={multi_bits}")
    OPT_INFO(msg)
    _mvn_correction(g, cdataloader, mode, alpha, beta, gamma,
                    act_bits, wgt_bits, bias_bits, lut_bits, multi_bits, mscopes)


def _mvn_correction(g, cdataloader, mode, alpha, beta, gamma, act_bits, wgt_bits, bias_bits, lut_bits, multi_bits,
                    mscopes):
    def scale2minmax(scale, is_signed, qrange):
        frange = qrange / scale
        if is_signed:
            return -1 * frange / 2,  frange / 2
        else:
            return 0, frange

    mvn_optypes = [OpType.MVN, OpType.LayerNorm, OpType.InstanceNorm, OpType.GroupNorm, OpType.RMSNorm]
    # ancestors_optypes need to save precision that mvn_optypes needed,
    # optypes like concat, cast can not truely increase precision when using higher quantization bits
    ancestors_optypes = mvn_optypes + [OpType.Convolution, OpType.ConvTranspose, OpType.Convolution3D, OpType.ConvTranspose3D,
                                       OpType.DepthwiseConv, OpType.FullyConnected,
                                       OpType.RNN, OpType.BasicLSTM, OpType.GRUv1, OpType.GRUv3,
                                       OpType.BatchNorm,  OpType.MatMul, OpType.Eltwise,
                                       OpType.Input, OpType.Constant]
    from AIPUBuilder.Optimizer.logger import tqdm
    with tqdm(total=len(g.nodes), desc='mvn_correction', file=sys.stdout, leave=True) as pbar:
        for n in g.nodes:
            if n.type in mvn_optypes and mscopes.get(n):
                ti = n.inputs[0]
                pn = n.parents[0]
                while True:
                    if pn.type in ancestors_optypes or len(pn.parents) < 1:
                        ti = pn.outputs[0]
                        break
                    else:
                        pn = pn.parents[0]
                to = n.outputs[0]
                s = ((ti.scale * to.scale) ** 0.5) * \
                    torch.max(torch_tensor(alpha, device=to.device), (ti.scale / to.scale) ** 0.5)
                if mode <= 0:
                    # care more about performance
                    pass
                else:
                    pnlist = [pn, ]
                    if (pn not in n.parents) or (beta > 0) or (gamma > 0):
                        for xn in g.nodes:
                            if (pn.attrs['tgid'] - beta) < xn.attrs['tgid'] < (n.attrs['tgid'] + gamma):
                                pnlist.append(xn)
                    pnlist.append(n)
                    for xn in pnlist:
                        xn.attrs['q_bits_activation'] = act_bits
                        xn.attrs['q_bits_weight'] = wgt_bits
                        xn.attrs['q_bits_bias'] = bias_bits
                        xn.attrs['lut_items_in_bits'] = lut_bits
                        q_mode_activation = xn.attrs["q_mode_activation"]
                        xn.attrs['q_mode_activation'] = QuantMode.to_symmetric(q_mode_activation)
                        xn.attrs['multiplier_bits'] = multi_bits
                        ti.qmin, ti.qmax = bits2range(16, is_signed(ti.dtype))
                if alpha > 0.0:
                    ti_min, ti_max = scale2minmax(s, is_signed(ti.dtype), ti.qmax - ti.qmin)
                    ti.min = torch_tensor(ti_min, device=ti.min.device)
                    ti.max = torch_tensor(ti_max, device=ti.min.device)
                else:
                    # not update, just skip
                    pass

            pbar.update(1)
        pbar.refresh()
