# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

inverse_sqrt_table = [32767, 31790, 30894, 30070, 29309, 28602, 27945, 27330, 26755, 26214, 25705, 25225, 24770, 24339, 23930,
                      23541, 23170, 22817, 22479, 22155, 21845, 21548, 21263, 20988, 20724, 20470, 20225, 19988, 19760, 19539,
                      19326, 19119, 18919, 18725, 18536, 18354, 18176, 18004, 17837, 17674, 17515, 17361, 17211, 17064, 16921,
                      16782, 16646, 16514, 16384]


def calculate_inverse_sqrt(x):
    def normalize_to31bit(var):
        var = var.to(torch.int64)
        var_out = torch.zeros_like(var)
        for k in range(0, 30):
            cond = var < 0x40000000
            var = torch.where(cond, var << 1, var)
            var_out = torch.where(cond, var_out + 1, var_out)
        var_out = 31 - var_out
        return var_out, var

    def normalised_inverse_sqrt(frac, exp):
        frac = torch.where(exp & 0x1 == 1, frac >> 1, frac)
        exp = (exp-1) >> 1
        frac = frac >> 9
        i = (frac >> 16) - 16
        i = torch.maximum(i, torch.zeros_like(i))
        frac = frac >> 1
        a = frac & 0x7fff
        isqrt = torch.tensor(inverse_sqrt_table, dtype=torch.int64, device=frac.device)
        tmp1 = torch.gather(isqrt, 0, i.reshape(-1))
        frac = tmp1 << 16
        tmp = tmp1 - torch.gather(isqrt, 0, (i+1).reshape(-1))
        frac = frac - (torch.multiply(tmp, a) << 1)
        return frac, exp
    vshape = x.shape
    vtype = x.dtype
    exp, x = normalize_to31bit(x.reshape(-1))
    x, exp = normalised_inverse_sqrt(x, exp)
    y = x >> exp
    return y.reshape(vshape).to(vtype)


def lookup_lut_powerof2(inputs, lut, lut_in_bits, in_is_signed, lut_out_bits, out_is_signed, align_mode=None):
    """the lut is size of 2^N, use the inputs high Nbits as the lut index and the low bits for interplate.
    :param inputs: inputs data as lut indexes
    :param lut: lookup table
    :param lut_in_bits: indexes range used bits
    :param in_is_signed: indexes sign
    :param lut_out_bits: lut value range used bits
    :param out_is_signed: lut value sign
    :param align_mode: right align and left align
    :return:
    """
    from AIPUBuilder.Optimizer.utils.dtype_utils import bits2range
    from AIPUBuilder.Optimizer.logger import OPT_FATAL, OPT_DEBUG
    import math
    table_size = lut.numel()
    table_size_bits = int(math.log2(table_size))
    if 2 ** table_size_bits != table_size:
        OPT_FATAL('lookup_lut_powerof2 function can only support lut which has 2**N items.')

    in_qmin, _ = bits2range(lut_in_bits, in_is_signed)
    out_qmin, out_qmax = bits2range(lut_out_bits, out_is_signed)
    lutp = torch.nn.functional.pad(lut, (0, 1), value=lut[-1])
    lut_long = lutp.to(torch.long)
    inputs_long = inputs.to(torch.long)
    indexes = (inputs_long - in_qmin)
    ibits = 0
    lut_index = indexes
    interp_index = 0
    if lut_in_bits > table_size_bits:
        ibits = lut_in_bits - table_size_bits
        if align_mode is not None:
            if align_mode == 'right_align':
                diff = indexes.max() - (table_size - 1)
                lidx = torch.clamp(indexes - diff, 0, table_size - 1)
                lut_index = lidx << ibits
            elif align_mode == 'left_align':
                diff = indexes.min() - 0
                lidx = torch.clamp(indexes - diff, 0, table_size - 1)
                lut_index = lidx << ibits
            else:
                OPT_DEBUG("lookup_lut_power2 align_mode=default.")
        lut_index = (indexes >> ibits).long()
        interp_index = indexes - (lut_index << ibits)
    inter_value = (lut_long[lut_index+1] - lut_long[lut_index]) * interp_index
    inter_value = inter_value >> ibits
    out = lut_long[lut_index] + inter_value
    out = torch.clamp(out, out_qmin, out_qmax).float()
    return out
