# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
import torch
from datetime import datetime
import time
import numpy as np
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.sigmoid import *
from AIPUBuilder.Optimizer.ops.tanh import *

rnn_tanh_lut_all_v3 = {}
rnn_sigm_lut_all_v3 = {}
rnn_tanh_lut_all_v1 = {}
rnn_sigm_lut_all_v1 = {}


rnn_tanh_lut_all_v3.update({'int8': {'table':
                                     [-127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -126, -126,
                                      -126, -126, -126, -125, -125, -125, -125, -125, -125, -125, -125, -124, -124, -124,
                                      -124, -124, -124, -123, -123, -123, -123, -123, -122, -122, -122, -122, -121, -121,
                                      -121, -121, -120, -120, -120, -119, -119, -119, -118, -118, -117, -117, -117, -116,
                                      -116, -115, -115, -114, -114, -113, -112, -112, -111, -110, -110, -109, -108, -108,
                                      -107, -106, -105, -104, -103, -102, -101, -100, -99, -98, -97, -96, -95, -94, -92,
                                      -91, -90, -88, -87, -86, -84, -83, -81, -79, -78, -76, -74, -72, -71, -69, -67, -65,
                                      -63, -61, -59, -57, -55, -52, -50, -48, -46, -43, -41, -39, -36, -34, -31, -29, -26,
                                      -24, -21, -18, -16, -13, -11, -8, -5, -3, 0, 3, 5, 8, 11, 13, 16, 18, 21, 24, 26, 29,
                                      31, 34, 36, 39, 41, 43, 46, 48, 50, 52, 55, 57, 59, 61, 63, 65, 67, 69, 71, 72, 74, 76,
                                      78, 79, 81, 83, 84, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
                                      104, 105, 106, 107, 108, 108, 109, 110, 110, 111, 112, 112, 113, 114, 114, 115, 115, 116,
                                      116, 117, 117, 117, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 121, 122, 122,
                                      122, 122, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125,
                                      125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127,
                                      127],
                                     'in_scale': 12364.9056604,  # 2.65
                                     'out_scale': 127.0
                                     }})

rnn_sigm_lut_all_v3.update({'int8': {'table':
                                     [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                                      3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9,
                                      9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17,
                                      18, 18, 19, 19, 20, 21, 21, 22, 22, 23, 24, 25, 25, 26, 27, 28, 28, 29, 30, 31,
                                      32, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                      52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73,
                                      74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
                                      95, 96, 96, 97, 98, 99, 100, 100, 101, 102, 103, 103, 104, 105, 106, 106, 107,
                                      107, 108, 109, 109, 110, 110, 111, 111, 112, 112, 113, 113, 114, 114, 115, 115,
                                      116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 120,
                                      121, 121, 121, 121, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 124,
                                      124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126,
                                      126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127,
                                      127, 127],
                                     'in_scale': 7123.2608695,  # 4.6
                                     'out_scale': 127.0
                                     }})

rnn_tanh_lut_all_v1.update({'int8': {'table':
                                     [-127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126,
                                      -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125,
                                      -125, -125, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -124,
                                      -123, -123, -123, -123, -123, -123, -122, -122, -122, -122, -121, -121, -121, -120,
                                      -120, -120, -119, -119, -119, -118, -118, -117, -117, -116, -116, -115, -115, -114,
                                      -113, -113, -112, -111, -111, -110, -109, -108, -107, -106, -106, -105, -103, -102,
                                      -101, -100, -99, -98, -96, -95, -94, -92, -91, -89, -87, -86, -84, -82, -80, -78, -76,
                                      -74, -72, -70, -68, -66, -63, -61, -58, -56, -53, -51, -48, -45, -42, -40, -37, -34,
                                      -31, -28, -25, -22, -19, -16, -13, -9, -6, -3, 0, 3, 6, 9, 13, 16, 19, 22, 25, 28,
                                      31, 34, 37, 40, 42, 45, 48, 51, 53, 56, 58, 61, 63, 66, 68, 70, 72, 74, 76, 78, 80,
                                      82, 84, 86, 87, 89, 91, 92, 94, 95, 96, 98, 99, 100, 101, 102, 103, 105, 106, 106,
                                      107, 108, 109, 110, 111, 111, 112, 113, 113, 114, 115, 115, 116, 116, 117, 117, 118,
                                      118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 122, 123, 123, 123,
                                      123, 123, 123, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125,
                                      125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
                                      126, 126, 126, 126, 126, 126, 126, 126, 127, 127],
                                     'in_scale': 128/3.1 * 256.,
                                     'out_scale': 127.0
                                     }})


rnn_sigm_lut_all_v1.update({'int8': {'table':
                                     [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                      2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4,
                                      5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11,
                                      11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20, 21, 22, 22,
                                      23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42,
                                      44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60, 62, 63, 64, 65, 66, 68,
                                      69, 71, 72, 74, 75, 76, 78, 79, 80, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93,
                                      94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 106, 107, 108,
                                      109, 109, 110, 111, 111, 112, 113, 113, 114, 114, 115, 115, 116, 116, 117,
                                      117, 117, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 121, 122,
                                      122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124,
                                      124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126,
                                      126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
                                      126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127],
                                     'in_scale': 128.0 / 6.2 * 256.0,
                                     'out_scale': 127.0
                                     }})

rnn_tanh_lut_all_v3.update({'int16': {'table':
                                      [-32767, -32767, -32767, -32766, -32766, -32766, -32766, -32766, -32766, -32766, -32766,
                                       -32766, -32766, -32766, -32766, -32765, -32765, -32765, -32765, -32765, -32764, -32764,
                                       -32764, -32764, -32763, -32763, -32762, -32762, -32761, -32761, -32760, -32760, -32759,
                                       -32758, -32757, -32756, -32755, -32754, -32753, -32751, -32750, -32748, -32746, -32744,
                                       -32742, -32740, -32737, -32734, -32731, -32727, -32723, -32719, -32714, -32709, -32703,
                                       -32697, -32690, -32683, -32675, -32666, -32656, -32645, -32633, -32619, -32605, -32589,
                                       -32572, -32553, -32531, -32508, -32483, -32455, -32425, -32391, -32355, -32315, -32270,
                                       -32222, -32169, -32111, -32047, -31977, -31900, -31817, -31725, -31624, -31514, -31393,
                                       -31261, -31117, -30959, -30787, -30599, -30393, -30169, -29925, -29659, -29369, -29054,
                                       -28712, -28340, -27937, -27501, -27029, -26518, -25968, -25375, -24738, -24053, -23320,
                                       -22537, -21701, -20812, -19868, -18869, -17815, -16706, -15542, -14325, -13058, -11742,
                                       -10382, -8980, -7542, -6073, -4578, -3063, -1535, 0, 1535, 3063, 4578, 6073, 7542, 8980,
                                       10382, 11742, 13058, 14325, 15542, 16706, 17815, 18869, 19868, 20812, 21701, 22537, 23320,
                                       24053, 24738, 25375, 25968, 26518, 27029, 27501, 27937, 28340, 28712, 29054, 29369, 29659,
                                       29925, 30169, 30393, 30599, 30787, 30959, 31117, 31261, 31393, 31514, 31624, 31725, 31817,
                                       31900, 31977, 32047, 32111, 32169, 32222, 32270, 32315, 32355, 32391, 32425, 32455, 32483,
                                       32508, 32531, 32553, 32572, 32589, 32605, 32619, 32633, 32645, 32656, 32666, 32675, 32683,
                                       32690, 32697, 32703, 32709, 32714, 32719, 32723, 32727, 32731, 32734, 32737, 32740, 32742,
                                       32744, 32746, 32748, 32750, 32751, 32753, 32754, 32755, 32756, 32757, 32758, 32759, 32760,
                                       32760, 32761, 32761, 32762, 32762, 32763, 32763, 32764, 32764, 32764, 32764, 32765, 32765,
                                       32765, 32765, 32765, 32766, 32766, 32766, 32766, 32766, 32766, 32766, 32766, 32766, 32766,
                                       32766, 32766, 32767, 32767, 32767],
                                      'in_scale': 32767/6.0,
                                      'out_scale': 32767.0
                                      }})

rnn_sigm_lut_all_v3.update({'int16': {'table':
                                      [1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10, 11, 12,
                                       13, 14, 15, 17, 18, 20, 21, 23, 25, 27, 29, 31, 34, 37, 40, 43, 46, 50, 54, 58, 63, 68,
                                       74, 80, 86, 93, 101, 109, 118, 127, 138, 149, 161, 174, 188, 203, 219, 237, 256, 277,
                                       299, 323, 349, 377, 407, 440, 475, 513, 554, 598, 646, 698, 753, 813, 877, 946, 1021,
                                       1101, 1187, 1280, 1379, 1486, 1601, 1724, 1856, 1998, 2150, 2312, 2486, 2671, 2869, 3081,
                                       3306, 3545, 3800, 4070, 4357, 4661, 4982, 5321, 5678, 6054, 6450, 6864, 7297, 7750, 8222,
                                       8712, 9221, 9747, 10291, 10850, 11424, 12012, 12613, 13224, 13845, 14473, 15107, 15744, 16384,
                                       17024, 17661, 18295, 18923, 19544, 20155, 20756, 21344, 21918, 22477, 23021, 23547, 24056,
                                       24546, 25018, 25471, 25904, 26318, 26714, 27090, 27447, 27786, 28107, 28411, 28698, 28968,
                                       29223, 29462, 29687, 29899, 30097, 30282, 30456, 30618, 30770, 30912, 31044, 31167, 31282,
                                       31389, 31488, 31581, 31667, 31747, 31822, 31891, 31955, 32015, 32070, 32122, 32170, 32214,
                                       32255, 32293, 32328, 32361, 32391, 32419, 32445, 32469, 32491, 32512, 32531, 32549, 32565,
                                       32580, 32594, 32607, 32619, 32630, 32641, 32650, 32659, 32667, 32675, 32682, 32688, 32694,
                                       32700, 32705, 32710, 32714, 32718, 32722, 32725, 32728, 32731, 32734, 32737, 32739, 32741,
                                       32743, 32745, 32747, 32748, 32750, 32751, 32753, 32754, 32755, 32756, 32757, 32758, 32758,
                                       32759, 32760, 32760, 32761, 32761, 32762, 32762, 32763, 32763, 32764, 32764, 32764, 32764,
                                       32765, 32765, 32765, 32765, 32766, 32766, 32766, 32766, 32766, 32766, 32767],
                                      'in_scale': 32768/10.0,
                                      'out_scale': 32767.0
                                      }})


def get_activation_scale(node):
    activations_list = node.get_param('activations') if 'activations' in node.params else ['SIGMOID', 'TANH', 'TANH']
    q_bits_activation = node.attrs["q_bits_activation"]
    dtype = dtype2str(bits2dtype(q_bits_activation, is_signed=True))
    scales = []
    for act in activations_list:
        # has lut
        act = act.upper()
        if g_rnn_activation_func[act][0]:
            in_scale = g_rnn_lut_supported[(node.type, act)][dtype]['in_scale']
            out_scale = g_rnn_lut_supported[(node.type, act)][dtype]['out_scale']
            scales.extend([in_scale, out_scale])
        #not lut
        else:  # no cover
            scales.extend([1, 1])
            OPT_WARN(
                'this activation[%s] donot have lut, need calculate in_scale and out_scale,default is [1,1]' % (act))
            pass
    return scales


def lookup_table(inputs, lut_table):
    index = torch.squeeze(inputs).long()
    index = index + 128
    out = torch.index_select(lut_table, 0, index)
    return torch.unsqueeze(out, dim=0)


def itp_lut_no_mirror(d, LUT_DATA):
    d = torch.squeeze(d).int()
    ori_d = d
    d = d.cpu().numpy()
    LUT_DATA = LUT_DATA.cpu().numpy()

    Global_SDP_LUT_ACCESS_DATA = 0
    Global_SDP_LUT_ACS_PTR = 0
    Global_SDP_LUT_BIAS = -(2 << 14)
    Global_SDP_LUT_PRSH = 0
    Global_SDP_LUT_MAX_SLOPE = 0
    Global_SDP_LUT_MIN_SLOPE = 0
    Global_SDP_LUT_MODE = 0

    one_64b = 1
    max_slope_8b = Global_SDP_LUT_MAX_SLOPE
    min_slope_8b = Global_SDP_LUT_MIN_SLOPE
    d_out = []

    def lookup_element(din, lut_data):
        lut_data = lut_data.astype(np.int64)
        lut_sub = din.astype(np.int64) - np.array(Global_SDP_LUT_BIAS, dtype=np.int64)
        lut_rsh = lut_sub // 2 ** Global_SDP_LUT_PRSH

        pos_out_of_range = lut_rsh > ((one_64b << 16) - 1)
        neg_out_of_range = lut_rsh < 0

        lut_rsh_rsh = lut_rsh // 2 ** 8
        lut_idx = lut_rsh_rsh & 0xff
        lut_delta = lut_rsh & 0xff

        pos_out_of_range_clamp_en = (lut_rsh - (one_64b << 16)) > ((one_64b << 16) - 1)
        neg_out_of_range_clamp_en = lut_rsh < (0 - ((one_64b << 16) - 1))

        def pos_out_of_range_clamp_en_true():
            return one_64b << 16 - 1

        def pos_out_of_range_clamp_en_false():
            return (lut_rsh - (one_64b << 16))

        def neg_out_of_range_delta_true():
            return (0 - ((one_64b << 16) - 1))

        def neg_out_of_range_delta_false():
            return lut_rsh

        pos_out_of_range_delta = pos_out_of_range_clamp_en_true() if pos_out_of_range_clamp_en \
            else pos_out_of_range_clamp_en_false()

        neg_out_of_range_delta = neg_out_of_range_delta_true() if neg_out_of_range_clamp_en \
            else neg_out_of_range_delta_false()

        def pos_out_of_range_true():
            return pos_out_of_range_delta * max_slope_8b + lut_data[256]

        def neg_out_of_range_true():
            return neg_out_of_range_delta * min_slope_8b + lut_data[0]

        def pos_and_neg_out_of_range_both_false():
            dout = (lut_data[lut_idx + 1] - lut_data[lut_idx]) * lut_delta + (lut_data[lut_idx] * 2 ** 8)
            dout = dout // 2 ** 8
            return dout

        def other_func():
            return neg_out_of_range_true() if neg_out_of_range else pos_and_neg_out_of_range_both_false()

        dout = pos_out_of_range_true() if pos_out_of_range else other_func()
        return dout

    output = np.zeros_like(d)
    for idx in range(d.shape[0]):
        output[idx] = lookup_element(d[idx], LUT_DATA)
    output = torch.tensor(output, device=ori_d.device)
    return torch.unsqueeze(output, dim=0).float()


def absorb_input_h_zp_to_bias(node, *args):
    inp_zerop = node.inputs[0].zerop
    # currently h_zerop always is 0, because h0 is symmetric quantization
    h_zerop = node.placeholders[0].zerop
    w = node.constants["weights"].betensor.permute(1, 0)
    b = node.constants["biases"]
    bias = b.betensor[:w.shape[1]]
    input_size = node.get_param('input_size')
    cell_size = node.get_param('cell_size')
    x_w = w[:input_size, :]
    h_w = w[input_size:, :]
    inp_zerop_mat = torch.full((1, input_size), inp_zerop[0]).to(node.inputs[0].device).to(x_w.dtype)
    h_zerop_mat = torch.full((1, cell_size), h_zerop[0]).to(node.inputs[0].device).to(h_w.dtype)
    x_weights_sum = torch.squeeze(torch.matmul(inp_zerop_mat, x_w))
    h_weights_sum = torch.squeeze(torch.matmul(h_zerop_mat, h_w))
    bias += (x_weights_sum + h_weights_sum)
    if 'version' in node.params and node.params['version'] == "GRUV1":
        hidden_bias = b.betensor[w.shape[1]:]
        bias = torch.cat((bias, hidden_bias), dim=0)
    b.betensor = bias.clamp(b.qmin, b.qmax)


gru_lut_activationsIdx_dict = {
    'lut_rt': 0,
    'lut_zt': 0,
    'lut_ht': 1
}

lstm_lut_activationsIdx_dict = {
    'lut_it': 0,
    'lut_ft': 0,
    'lut_ct': 1,
    'lut_ot': 0,
    'lut_h': 2
}

g_rnn_activation_clamp = {  # bit
                            8:  {  # clamp_min     #clamp_max
                                (OpType.RNN, 'SIGMOID'): (-4.564205,     4.564205),
                                (OpType.RNN, 'TANH'):    (-2.6293793,     2.6293793),
                                (OpType.GRUv3, 'SIGMOID'): (-4.8,   4.8),
                                (OpType.GRUv3, 'TANH'):    (-2.8,     2.8),
                                (OpType.BasicLSTM, 'SIGMOID'): (-4.564205,     4.564205),
                                (OpType.BasicLSTM, 'TANH'):    (-2.6293793,     2.6293793),
                                (OpType.GRUv1, 'SIGMOID'): (-4.564202,     4.564202),
                                (OpType.GRUv1, 'TANH'):    (-2.629377,     2.629377),
                            },
    16:
    {  # clamp_min     #clamp_max
                                (OpType.RNN, 'SIGMOID'): (-10,     10),
                                (OpType.RNN, 'TANH'):    (-6.0,    6.0),
                                (OpType.GRUv3, 'SIGMOID'): (-10,     10),
                                (OpType.GRUv3, 'TANH'):    (-6.0,    6.0),
                                (OpType.BasicLSTM, 'SIGMOID'): (-10.0,     10.0),
                                (OpType.BasicLSTM, 'TANH'):    (-6.0,    6.0),
                                (OpType.GRUv1, 'SIGMOID'): (-10,     10),
                                (OpType.GRUv1, 'TANH'):    (-6.0,    6.0),
                            }
}


g_rnn_activation_func = {  # (activation) : (has_lut,  float_forward,  quantize_forward,  16bit_forward)
    'SIGMOID':      (True,      torch.sigmoid, lookup_table,      itp_lut_no_mirror, sigmoid_quantize),
    'TANH':         (True,       torch.tanh,   lookup_table,      itp_lut_no_mirror, tanh_quantize),
}

g_rnn_lut_supported = {   # (layer_type, activation) : lut
                          (OpType.BasicLSTM, 'SIGMOID'): rnn_sigm_lut_all_v3,
                          (OpType.BasicLSTM, 'TANH'): rnn_tanh_lut_all_v3,
                          (OpType.GRUv3, 'SIGMOID'): rnn_sigm_lut_all_v3,
                          (OpType.GRUv3, 'TANH'): rnn_tanh_lut_all_v3,
                          (OpType.GRUv1, 'SIGMOID'): rnn_sigm_lut_all_v1,
                          (OpType.GRUv1, 'TANH'): rnn_tanh_lut_all_v1,
}


def get_bak_tensor_property(t):
    _tensor_default_property = get_tensor_default_property()
    bak_tensor_property = dict()
    property = {}
    for p in _tensor_default_property:
        property.update({p: t.__getattribute__(p)})
    bak_tensor_property.update({t.name: property})
    return bak_tensor_property


def generate_activation_lut(node, activation, new_input, new_output, *args):
    _tensor_default_property = get_tensor_default_property()
    for p in _tensor_default_property:
        node.inputs[0].__setattr__(p, new_input.__getattribute__(p))
        node.outputs[0].__setattr__(p, new_output.__getattribute__(p))
    q_mode_activation_bak = node.attrs["q_mode_activation"]
    force_dtype_int = node.force_dtype_int
    node.attrs["q_mode_activation"] = "per_tensor_symmetric_restricted_range"
    # currently only support lut[256] for basiclstm/gru
    node.attrs['lut_items_in_bits'] = 8

    # call activication op from AIPUBuilder/Optimizer/ops/...
    node.force_dtype_int = True
    g_rnn_activation_func[activation][4](node, *args)
    lut = node.constants.pop('lut')

    for p in _tensor_default_property:
        new_output.__setattr__(p, node.outputs[0].__getattribute__(p))
    node.attrs["q_mode_activation"] = q_mode_activation_bak
    node.force_dtype_int = force_dtype_int
    node.quantized = False

    return lut


def generate_lut_with_placeholders(node, activation_idx, activation, q_mode_activation, q_bits_activation, activation_idx_lut_name, with_lut=False, with_clamp=False, *args):
    placehoder_idx = 1 + activation_idx * 2

    ph_in = node.placeholders[placehoder_idx]

    if with_clamp:
        clip_min, clip_max = g_rnn_activation_clamp[q_bits_activation][(node.type, activation)]
        ph_in.max = ph_in.max if ph_in.max <= clip_max else clip_max
        ph_in.min = ph_in.min if ph_in.min >= clip_min else clip_min

    ph_in.scale, ph_in.zerop, ph_in.qmin, ph_in.qmax, ph_in.dtype = \
        get_linear_quant_params_from_tensor(
            ph_in, "per_tensor_symmetric_restricted_range", q_bits_activation, is_signed=True)
    ph_in.qbits = q_bits_activation
    ph_in.qinvariant = False
    in_scale = ph_in.scale

    ph_out = node.placeholders[placehoder_idx+1]
    ph_out.scale, ph_out.zerop, ph_out.qmin, ph_out.qmax, ph_out.dtype = \
        get_linear_quant_params_from_tensor(
            ph_out, "per_tensor_symmetric_restricted_range", q_bits_activation, is_signed=True)
    ph_out.qbits = q_bits_activation
    ph_out.qinvariant = False
    out_scale = ph_out.scale

    if with_clamp:
        length = 1
        for s in ph_in.ir_shape:
            length *= s
        lut_in = torch.linspace(ph_in.min.item(), ph_in.max.item(), steps=length)
        lut_out = g_rnn_activation_func[activation][1](lut_in)
        ph_out.min = lut_out.min()
        ph_out.max = lut_out.max()

    if with_lut:
        lut = generate_activation_lut(node, activation, ph_in, ph_out, *args)
        for lut_name in activation_idx_lut_name[activation_idx]:
            node.constants[lut_name] = lut

    return in_scale, out_scale


def compress_int32_to_int16(array, lmin, lmax, pre_shift=0):
    import math
    bmin = min(array.min().item(), -1)
    bmax = max(array.max().item(), 1)
    lbits = math.ceil(max(math.log2(bmax / lmax), math.log2(bmin / lmin)))
    if lbits > pre_shift:
        array = ((array.long() >> lbits) << lbits).long()
    else:
        array = ((array.long() >> pre_shift) << pre_shift).long()
    return array
