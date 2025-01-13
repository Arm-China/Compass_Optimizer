# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import torch
from typing import Union

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
table_Log2_q15 = [0,   184,   368,   551,   733,   914,  1095,  1275,  1455,  1633,
                  1811,  1989,  2166,  2342,  2517,  2692,  2866,  3039,  3212,  3385,
                  3556,  3727,  3897,  4067,  4236,  4405,  4573,  4740,  4907,  5073,
                  5239,  5404,  5568,  5732,  5895,  6058,  6220,  6382,  6543,  6703,
                  6863,  7023,  7182,  7340,  7498,  7655,  7812,  7968,  8124,  8279,
                  8434,  8588,  8742,  8895,  9048,  9200,  9352,  9503,  9654,  9804,
                  9954, 10104, 10253, 10401, 10549, 10696, 10843, 10990, 11136, 11282,
                  11427, 11572, 11716, 11860, 12004, 12147, 12289, 12431, 12573, 12715,
                  12855, 12996, 13136, 13276, 13415, 13554, 13692, 13830, 13968, 14105,
                  14242, 14378, 14514, 14650, 14785, 14920, 15055, 15189, 15322, 15456,
                  15589, 15721, 15854, 15986, 16117, 16248, 16379, 16509, 16639, 16769,
                  16898, 17027, 17156, 17284, 17412, 17540, 17667, 17794, 17921, 18047,
                  18173, 18298, 18424, 18548, 18673, 18797, 18921, 19045, 19168, 19291,
                  19414, 19536, 19658, 19780, 19901, 20022, 20143, 20263, 20383, 20503,
                  20623, 20742, 20861, 20980, 21098, 21216, 21334, 21451, 21568, 21685,
                  21802, 21918, 22034, 22150, 22265, 22380, 22495, 22610, 22724, 22838,
                  22952, 23066, 23179, 23292, 23404, 23517, 23629, 23741, 23852, 23964,
                  24075, 24186, 24296, 24407, 24517, 24627, 24736, 24845, 24955, 25063,
                  25172, 25280, 25388, 25496, 25604, 25711, 25818, 25925, 26031, 26138,
                  26244, 26350, 26455, 26561, 26666, 26771, 26876, 26980, 27084, 27188,
                  27292, 27396, 27499, 27602, 27705, 27808, 27910, 28012, 28114, 28216,
                  28318, 28419, 28520, 28621, 28722, 28822, 28922, 29022, 29122, 29222,
                  29321, 29421, 29520, 29618, 29717, 29815, 29914, 30012, 30109, 30207,
                  30304, 30401, 30498, 30595, 30692, 30788, 30884, 30980, 31076, 31172,
                  31267, 31362, 31457, 31552, 31647, 31741, 31836, 31930, 32024, 32117,
                  32211, 32304, 32397, 32490, 32583, 32676]
# table_Log2_Q31 =[          0,    95335645,   187825021,   277633165,   364911162,
#          449797678,   532420281,   612896598,   691335320,   767837083,
#          842495250,   915396590,   986621888,  1056246482,  1124340739,
#         1190970490,  1256197405,  1320079339,  1382670639,  1444022426,
#         1504182841,  1563197273,  1621108567,  1677957208,  1733781493,
#         1788617686,  1842500157,  1895461516,  1947532725,  1998743213,
#         2049120974,  2098692655, 2147483647]

table_pow2_q15 = [
    32768, 32857, 32946, 33035, 33125, 33215, 33305, 33395, 33486, 33576,
    33667, 33759, 33850, 33942, 34034, 34126, 34219, 34312, 34405, 34498,
    34591, 34685, 34779, 34874, 34968, 35063, 35158, 35253, 35349, 35445,
    35541, 35637, 35734, 35831, 35928, 36025, 36123, 36221, 36319, 36417,
    36516, 36615, 36715, 36814, 36914, 37014, 37114, 37215, 37316, 37417,
    37518, 37620, 37722, 37824, 37927, 38030, 38133, 38236, 38340, 38444,
    38548, 38653, 38757, 38863, 38968, 39074, 39180, 39286, 39392, 39499,
    39606, 39714, 39821, 39929, 40037, 40146, 40255, 40364, 40473, 40583,
    40693, 40804, 40914, 41025, 41136, 41248, 41360, 41472, 41584, 41697,
    41810, 41923, 42037, 42151, 42265, 42380, 42495, 42610, 42726, 42841,
    42958, 43074, 43191, 43308, 43425, 43543, 43661, 43780, 43898, 44017,
    44137, 44256, 44376, 44497, 44617, 44738, 44859, 44981, 45103, 45225,
    45348, 45471, 45594, 45718, 45842, 45966, 46091, 46216, 46341, 46467,
    46593, 46719, 46846, 46973, 47100, 47228, 47356, 47484, 47613, 47742,
    47871, 48001, 48131, 48262, 48393, 48524, 48655, 48787, 48920, 49052,
    49185, 49319, 49452, 49586, 49721, 49856, 49991, 50126, 50262, 50399,
    50535, 50672, 50810, 50947, 51085, 51224, 51363, 51502, 51642, 51782,
    51922, 52063, 52204, 52346, 52488, 52630, 52773, 52916, 53059, 53203,
    53347, 53492, 53637, 53782, 53928, 54074, 54221, 54368, 54515, 54663,
    54811, 54960, 55109, 55258, 55408, 55558, 55709, 55860, 56012, 56163,
    56316, 56468, 56622, 56775, 56929, 57083, 57238, 57393, 57549, 57705,
    57861, 58018, 58176, 58333, 58491, 58650, 58809, 58968, 59128, 59289,
    59449, 59611, 59772, 59934, 60097, 60260, 60423, 60587, 60751, 60916,
    61081, 61247, 61413, 61579, 61746, 61914, 62081, 62250, 62419, 62588,
    62757, 62928, 63098, 63269, 63441, 63613, 63785, 63958, 64132, 64306,
    64480, 64655, 64830, 65006, 65182, 65359

]


def Log2_norm(tx, loginQ, logoutQ):
    vshape = tx.shape
    x = tx.reshape(-1)
    var_out = torch.zeros_like(x)
    # search counting the leading zero of the value with asm instruction(clz)
    for k in range(0, 30):
        cond = x < 0x40000000
        var_out = torch.where(cond, var_out+1, var_out)
        x = torch.where(cond, x << 1, x)

    i = ((x) >> 22) & 0xff
    # x = (x>> 9)
    # frac =  (x&0x7fff)
    table_Log2_norm = torch.tensor(table_Log2_q15, dtype=torch.int64, device=x.device)
    tmp0 = torch.gather(table_Log2_norm, 0, i.reshape(-1))
    tmp0 = tmp0 << 15
    y = (tmp0) >> (30-logoutQ)
    # tmp1 = torch.gather(table_Log2_norm, 0, (i+1).reshape(-1))
    # diff = (tmp1-tmp0)>>15
    # y =  (tmp0+(torch.multiply(diff,frac)))>>(31-logoutQ)
    y = y-((var_out-(30-loginQ)) << logoutQ)

    return y.reshape(vshape)


def simple_log2(tx, loginQ, logoutQ):
    vshape = tx.shape
    x = tx.reshape(-1)
    var_out = torch.zeros_like(x)
    # search counting the leading zero of the value with asm instruction(clz)
    for k in range(0, 30):
        cond = x < 0x40000000
        var_out = torch.where(cond, var_out+1, var_out)
        x = torch.where(cond, x << 1, x)
    if logoutQ == 0:
        y = torch.zeros_like(x)
    else:
        y = (x > 1518500250).int()
    y = y-((var_out-(30-loginQ)) << logoutQ)

    return y.reshape(vshape)
    # input tx is 16bit negative data, pow2inQ


def Pow2(tx, pow2inQ, logoutQ):
    vshape = tx.shape
    x = tx.reshape(-1)

    exponant = (x >> pow2inQ)
    i = (x & ((1 << pow2inQ)-1)).long() >> (pow2inQ-8)

    table_pow2 = torch.tensor(table_pow2_q15, dtype=torch.int32, device=x.device)
    tmp0 = torch.gather(table_pow2, 0, i.reshape(-1))

    tmp0 = tmp0 << 15
    y = tmp0 >> (30-logoutQ-exponant)
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


def lookup_float_index_lut(t: torch.Tensor, lut: torch.Tensor, index_scale: float, index_offset: Union[float, int], mirror_mode: bool = False, value_offset_for_mirror_mode: Union[float, int] = 0.0) -> torch.Tensor:
    def lookup_lut(x):
        y = x
        table = lut.flatten()
        table_size = table.numel()
        left_slope = table[1] - table[0]
        right_slope = table[-1] - table[-2]
        left_mask = x < 0
        right_mask = x > (table_size-1)

        y_left = torch.where(left_mask, table[0] - left_slope * (0 - x), x)
        y_right = torch.where(right_mask, table[-1] + right_slope * (x - (table_size-1)), x)
        index = torch.clamp(x, 0, table_size-1).floor().long()
        lutp = torch.nn.functional.pad(table, (0, 1), value=table[-1])
        y_middle = lutp[index] + (lutp[index+1] - lutp[index]) * (x - index)
        y = torch.where(left_mask, y_left, y_middle)
        y = torch.where(right_mask, y_right, y)
        return y
    x_t = t * index_scale - index_offset
    if mirror_mode:
        y_pos = lookup_lut(x_t)
        y_neg = lookup_lut(x_t.negative()).negative()
        return torch.where(x_t >= 0.0, y_pos, y_neg) - value_offset_for_mirror_mode
    else:
        return lookup_lut(x_t)


def broadcasting_transform(x0, x1):
    from AIPUBuilder.Optimizer.logger import OPT_WARN
    # align axis
    x0_shape, x1_shape = x0.shape, x1.shape
    max_len = max(len(x0_shape), len(x1_shape))
    tile_a, tile_b = [1] * max_len, [1] * max_len
    x0, x1 = x0.tile(tile_a), x1.tile(tile_b)

    # check broadcast params
    for i in range(max_len):
        local_axis = max_len - i - 1
        if x0.shape[local_axis] == 1 and x1.shape[local_axis] % x0.shape[local_axis] == 0:
            tile_a[local_axis] = x1.shape[local_axis]
        elif x1.shape[local_axis] == 1 and x0.shape[i] % x1.shape[local_axis] == 0:
            tile_b[local_axis] = x0.shape[local_axis]
        elif x1.shape[local_axis] % x0.shape[local_axis] == 0:
            tile_a[local_axis] = int(x1.shape[local_axis] / x0.shape[local_axis])
        elif x0.shape[local_axis] % x1.shape[local_axis] == 0:
            tile_b[local_axis] = int(x0.shape[local_axis] / x1.shape[local_axis])
        elif x1.shape[local_axis] % x0.shape[local_axis] != 0 or x0.shape[local_axis] % x1.shape[local_axis] != 0:
            OPT_WARN('tensors are non-broadcastable')
    x0, x1 = x0.tile(tile_a), x1.tile(tile_b)

    return x0, x1


def x3_aiff_exp_approximation(f_vdata: torch.Tensor, pow2_f_lut: torch.Tensor) -> torch.Tensor:
    f_vdata[f_vdata.isnan()] = 0
    f_vdata = torch.clamp(f_vdata.float(), torch.finfo(torch.float32).min, torch.finfo(torch.float32).max)
    mantisa_bit = 16
    #lut_bits = 9
    vshape = f_vdata.shape
    # limit input
    f_vdata = (f_vdata < -126.9)*-126.9+f_vdata*(f_vdata >= -126.9)
    f_vdata = (f_vdata > 126.9)*126.9+f_vdata*(f_vdata < 126.9)

    exponent = f_vdata.floor()
    mantisa = ((f_vdata - exponent)*((1 << mantisa_bit)-1)).long()
    index = mantisa >> 7

    interp_bit = mantisa & 0x7f

    torch_pow2_f_lut = pow2_f_lut.to(f_vdata.device)
    diff = torch_pow2_f_lut[index+1] - torch_pow2_f_lut[index]
    post_lut = torch_pow2_f_lut[index] + (diff*interp_bit)*(2.0**-7)

    pow2_factor = post_lut*(2.0**exponent)
    pow2_fp24 = ((pow2_factor.to(torch.float32)).view(torch.int32) & 0xFFFFFF00).view(torch.float32)

    return pow2_fp24.reshape(vshape)


def x3_aiff_softmax_approximation(vx: torch.Tensor, axis, pow2_f_lut: torch.Tensor) -> torch.Tensor:
    vx[vx.isnan()] = 0
    vx = torch.clamp(vx.float(), torch.finfo(torch.float32).min, torch.finfo(torch.float32).max)
    max_v, _ = vx.max(axis, keepdim=True)
    f_vdata = (vx-max_v)*1.442695
    yy = x3_aiff_exp_approximation(f_vdata, pow2_f_lut)
    y_sum = yy.sum(axis, keepdim=True)
    # convert to fp24
    score = ((1/y_sum).view(torch.int) & 0xFFFFff00).view(torch.float32)
    # convert yy to fp16
    yy16 = yy.half()
    score = yy16*score
    # convert f32 softmax result to fp16 output
    f16 = score.half()
    return f16.reshape(vx.shape)
