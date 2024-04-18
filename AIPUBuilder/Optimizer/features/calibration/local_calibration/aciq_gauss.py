# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from . aciq_laplace import aciq_clipper
import torch


def aciq_gauss_calibration(t, *args):
    cstrategy = args[0]
    quantize_method = args[1]
    try:
        g = float(cstrategy[:-10])
    except:
        g = 1.0
    aciq_clipper(t, quantize_method, clipping_type=1, gamma=g)


# def aciq_clipper(t, quantize_method, clipping_type=0, gamma=1.0):
#     """
#     Implemented according to https://arxiv.org/pdf/1810.05723.pdf
#     use sympy to solve eq
#     """

#     alpha_laplace = {0: 1.05, 1: 1.86, 2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89,
#                      9: 11.16, 10: 12.44, 11: 13.73, 12: 15.02, 13: 16.33, 14: 17.64, 15: 18.95, 16: 20.27}
#     alpha_laplace_positive = {0: 1.86, 1: 2.83, 2: 3.89, 3: 5.03, 4: 6.2, 5: 7.41, 6: 8.64, 7: 9.89,
#                               8: 11.16, 9: 12.44, 10: 13.73, 11: 15.02, 12: 16.33, 13: 17.64, 14: 18.95, 15: 20.27, 16: 21.59}
#     alpha_gauss = {1: 1.24, 2: 1.71, 3: 2.15, 4: 2.55, 5: 2.93, 6: 3.28, 7: 3.61, 8: 3.92,
#                    9: 4.21, 10: 4.49, 11: 4.75, 12: 4.99, 13: 5.24, 14: 5.47, 15: 5.69, 16: 5.89}
#     alpha_gauss_positive = {1: 1.71, 2: 2.15, 3: 2.55, 4: 2.93, 5: 3.28, 6: 3.61, 7: 3.92,
#                             8: 4.21, 9: 4.49, 10: 4.75, 11: 4.99, 12: 5.24, 13: 5.47, 14: 5.69, 15: 5.89, 16: 6.04}

#     def get_alpha(half_range):
#         alpha_dict = {}
#         if 0 == clipping_type:
#             alpha_dict = alpha_laplace_positive if half_range else alpha_laplace
#         else:
#             alpha_dict = alpha_gauss_positive if half_range else alpha_gauss
#         alpha_const = alpha_dict[t.qbits] if t.qbits in alpha_dict else alpha_dict[8]
#         alpha = 0.0
#         alpha_key_axis = None
#         if 0 == clipping_type:
#             alpha = alpha_const * t.running_mad
#             if None != t.running_min_key_axis:
#                 alpha_key_axis = alpha_const * t.running_mad_key_axis
#         else:
#             alpha = alpha_const * t.running_std
#             if None != t.running_min_key_axis:
#                 alpha_key_axis = alpha_const * t.running_std_key_axis
#         return alpha, alpha_key_axis

#     if QuantMode.is_symmetric(quantize_method):
#         alpha, alpha_key_axis = get_alpha(False)
#         cvalue = abs(t.running_mean) + alpha * gamma
#         t.min = min(0.0, -cvalue)
#         t.max = max(0.0, cvalue)
#         if None != t.running_min_key_axis:
#             cvalue_key_axis = torch.abs(t.running_mean_key_axis) + alpha_key_axis * gamma
#             t.min_key_axis = torch.min(torch.zeros_like(t.running_min_key_axis), -1 * cvalue_key_axis)
#             t.max_key_axis = torch.max(torch.zeros_like(t.running_min_key_axis), cvalue_key_axis)
#     else:
#         half_range = False  # t.extrema_min >= 0
#         alpha, alpha_key_axis = get_alpha(half_range)
#         min_val = max(t.running_min, t.running_mean - alpha)
#         delta = alpha if half_range else 2 * alpha
#         t.min = min(0.0, min_val)
#         t.max = max(0.0, min_val + delta * gamma)
#         if None != t.running_min_key_axis:
#             min_val_key_axis = torch.max(t.running_min_key_axis, t.running_mean_key_axis - alpha_key_axis)
#             delta_key_axis = alpha_key_axis if half_range else 2 * alpha_key_axis
#             t.min_key_axis = torch.min(torch.zeros_like(t.running_min_key_axis), min_val_key_axis)
#             t.max_key_axis = torch.max(torch.zeros_like(t.running_min_key_axis),
#                                        min_val_key_axis + delta_key_axis * gamma)
