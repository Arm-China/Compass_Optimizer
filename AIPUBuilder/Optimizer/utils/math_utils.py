# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

import torch

inverse_sqrt_table = [32767, 32515, 32268, 32026, 31790, 31558, 31332, 31111, 30894,
                      30682, 30474, 30270, 30070, 29874, 29682, 29494, 29309, 29127,
                      28949, 28774, 28602, 28434, 28268, 28105, 27945, 27787, 27632,
                      27480, 27330, 27183, 27038, 26895, 26755, 26617, 26481, 26346,
                      26214, 26084, 25956, 25830, 25705, 25583, 25462, 25342, 25225,
                      25109, 24994, 24882, 24770, 24660, 24552, 24445, 24339, 24235,
                      24132, 24031, 23930, 23831, 23733, 23637, 23541, 23447, 23354,
                      23262, 23170, 23080, 22992, 22904, 22817, 22731, 22646, 22562,
                      22479, 22396, 22315, 22235, 22155, 22077, 21999, 21922, 21845,
                      21770, 21695, 21621, 21548, 21476, 21404, 21333, 21263, 21193,
                      21124, 21056, 20988, 20921, 20855, 20789, 20724, 20660, 20596,
                      20533, 20470, 20408, 20346, 20285, 20225, 20165, 20106, 20047,
                      19988, 19930, 19873, 19816, 19760, 19704, 19649, 19594, 19539,
                      19485, 19431, 19378, 19326, 19273, 19221, 19170, 19119, 19068,
                      19018, 18968, 18919, 18870, 18821, 18773, 18725, 18677, 18630,
                      18583, 18536, 18490, 18444, 18399, 18354, 18309, 18264, 18220,
                      18176, 18133, 18090, 18047, 18004, 17962, 17920, 17878, 17837,
                      17795, 17755, 17714, 17674, 17634, 17594, 17554, 17515, 17476,
                      17438, 17399, 17361, 17323, 17285, 17248, 17211, 17174, 17137,
                      17100, 17064, 17028, 16992, 16957, 16921, 16886, 16851, 16817,
                      16782, 16748, 16714, 16680, 16646, 16613, 16579, 16546, 16514,
                      16481, 16448, 16416, 16384]
# 32bit  data
table_Log2_Q31 = [0,    48034513,    95335645,   141925456,   187825021,
                  233054496,   277633165,   321579490,   364911162,   407645136,
                  449797678,   491384396,   532420281,   572919734,   612896598,
                  652364189,   691335320,   729822324,   767837083,   805391046,
                  842495250,   879160341,   915396590,   951213914,   986621888,
                  1021629764,  1056246482,  1090480686,  1124340739,  1157834731,
                  1190970490,  1223755601,  1256197405,  1288303019,  1320079339,
                  1351533050,  1382670639,  1413498396,  1444022426,  1474248656,
                  1504182841,  1533830570,  1563197273,  1592288229,  1621108567,
                  1649663276,  1677957208,  1705995083,  1733781493,  1761320910,
                  1788617686,  1815676059,  1842500157,  1869094003,  1895461516,
                  1921606515,  1947532725,  1973243777,  1998743213,  2024034488,
                  2049120974,  2074005959,  2098692655,  2123184198, 2147483647]
# table_Log2_Q31 =[          0,    95335645,   187825021,   277633165,   364911162,
#          449797678,   532420281,   612896598,   691335320,   767837083,
#          842495250,   915396590,   986621888,  1056246482,  1124340739,
#         1190970490,  1256197405,  1320079339,  1382670639,  1444022426,
#         1504182841,  1563197273,  1621108567,  1677957208,  1733781493,
#         1788617686,  1842500157,  1895461516,  1947532725,  1998743213,
#         2049120974,  2098692655, 2147483647]


def Log2_norm(tx, loginQ, logoutQ):
    vshape = tx.shape
    x = tx.reshape(-1)
    var_out = torch.zeros_like(x)
    # search counting the leading zero of the value with asm instruction(clz)
    for k in range(0, 30):
        cond = x < 0x40000000
        var_out = torch.where(cond, var_out+1, var_out)
        x = torch.where(cond, x << 1, x)

    i = ((x & 0xff000000) >> 24) & 0x3f
    x = (x >> 9)
    frac = (x & 0x7fff)
    table_Log2_norm = torch.tensor(table_Log2_Q31, dtype=torch.int64, device=x.device)
    tmp0 = torch.gather(table_Log2_norm, 0, i.reshape(-1))
    tmp1 = torch.gather(table_Log2_norm, 0, (i+1).reshape(-1))
    diff = (tmp1-tmp0) >> 15
    y = (tmp0+(torch.multiply(diff, frac))) >> (31-logoutQ)
    y = y-((var_out-(30-loginQ)) << logoutQ)

    return y.reshape(vshape)


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
        frac = frac >> 7
        i = (frac >> 16) - 64
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


def lookup_lut_powerof2(inputs, lut, lut_in_bits, in_is_signed, lut_out_bits, out_is_signed, is_float=False, align_mode=None):
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
    lut_long = lutp.to(torch.long) if not is_float else lutp.to(torch.float)
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
    if not is_float:
        inter_value = inter_value >> ibits
        out = lut_long[lut_index] + inter_value
        out = torch.clamp(out, out_qmin, out_qmax).long()
    else:
        inter_value = inter_value / (2**ibits)
        out = lut_long[lut_index] + inter_value
    return out


def broadcasting_transform(x0, x1):
    from AIPUBuilder.Optimizer.logger import OPT_WARN
    # align axis
    x0_shape, x1_shape = x0.shape, x1.shape
    max_len = max(len(x0_shape), len(x1_shape))
    tile_a, tile_b = [1 for c in range(max_len)], [1 for c in range(max_len)]
    x0, x1 = x0.tile(tile_a), x1.tile(tile_b)

    # check broadcast params
    for i in range(max_len):
        local_axis = max_len - i - 1
        if (x0.shape[local_axis] == 1 and x1.shape[local_axis] % x0.shape[local_axis] == 0):
            tile_a[local_axis] = x1.shape[local_axis]
        elif (x1.shape[local_axis] == 1 and x0.shape[i] % x1.shape[local_axis] == 0):
            tile_b[local_axis] = x0.shape[local_axis]
        elif x1.shape[local_axis] % x0.shape[local_axis] != 0 or x0.shape[local_axis] % x1.shape[local_axis] != 0:
            OPT_WARN('tensors are non-broadcastable')
    x0, x1 = x0.tile(tile_a), x1.tile(tile_b)
    return x0, x1
