# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import os
import torch
from torch import nn
from torch.autograd import Function
from sympy import false
import numpy as np
import math

from AIPUBuilder.Optimizer.utils import (
    torch_tensor,
    QuantMode,
    dtype2range,
    bits2range,
    bits2dtype,
    range2dtype,
    get_scale_approximation_params,
    linear_quantize_clip,
    is_float)
from AIPUBuilder.Optimizer.utils import is_signed as dtype_is_signed

from ..qatlogger import QAT_ERROR, QAT_INFO, QAT_DEBUG, QAT_WARN
from ..qinfo import QInfo, QuantStage, QScheme, CMode


def check_args(func):
    def wrapper(*args, **kwargs):
        '''
        add some check rules at here, like you want check whether args or kwargs has
        some parameters or attributes, or just print its, like:
            # print(f"check_args: len(args)={len(args)}, len(kwargs)={len(kwargs)}")
        '''
        out = func(*args, **kwargs)
        return out
    return wrapper


class Round(Function):
    @staticmethod
    def forward(ctx, input):
        result = torch.round(input)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return torch.clone(grad_output)


class Clamp(Function):
    @staticmethod
    def forward(ctx, input, clip_min, clip_max):
        result = torch.clamp(input, clip_min, clip_max)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return torch.clone(grad_output), None, None


class QBaseOperator(nn.Module):
    def __init__(self, dtype=None, name='') -> None:
        super(QBaseOperator, self).__init__()
        self._is_leaf_module = True
        self._use_input_QConfig = False  # use input QConfig for output if True.
        self._is_module_params_set = False
        self._is_eval_bypass = False
        self._input_shape = None
        self._output_shape = None
        self.name = name

        self.dtype = dtype
        self.quant_stage = QuantStage.FP32

        self.running_moment = 0.99
        self.prev_modules = []

    def set_prev_modules(self, prev_m):
        if len(self.prev_modules):
            QAT_WARN(f"{self.name} has setted prev modules")
            return
        a = prev_m
        if not isinstance(a, (tuple, list)):
            a = [a, ]
        self.prev_modules = a

    def serialize(self, args, **kwargs):
        pass

    def update_tensor_quantization(self, t, t_qinfo):
        from AIPUBuilder.core import Dtype
        '''
        finetune_forward --> finetune_backward -->evaluate_loop
        if evaluate_loop acc is ok, will serialie the IR.
        but after finetune_backward, the weight is not update its qinfos, so when serializing,
        re-statistic and re-calculate the qinfos.
        '''
        self.statistic(t.betensor, t_qinfo)
        _ = self.linear_affine(t_qinfo)

        t.quantization.scales = t_qinfo.scale.tolist()
        t.quantization.offsets = t_qinfo.zerop.tolist()

        if t_qinfo.dtype is not None:
            tdtype = t_qinfo.dtype
        else:
            tdtype = bits2dtype(t_qinfo.bits, True)
        t.quantization.aipu_dtype = Dtype(tdtype.name.lower())
        t.quantization.bits = t_qinfo.bits
        # t.quantization.mins = t_qinfo.mins.tolist() if t_qinfo.mins is not None else []
        # t.quantization.maxs = t_qinfo.maxs.tolist() if t_qinfo.maxs is not None else []
        if t_qinfo.dtype is not None:
            qmin, qmax = dtype2range(t_qinfo.dtype)
        # t.quantization.mins = [qmin]
        # t.quantization.maxs = [qmax]

    def get_quantization(self, t_qinfo):
        if t_qinfo.is_quantized:
            from AIPUBuilder.core import Quantization, Dtype, Scheme, QuantizedRange, Granularity
            q = Quantization()
            # return q
            q.scales = t_qinfo.scale.tolist()
            q.offsets = t_qinfo.zerop.tolist()
            if t_qinfo.dtype is not None:
                tdtype = t_qinfo.dtype
            else:
                tdtype = bits2dtype(t_qinfo.bits, True)
            q.aipu_dtype = Dtype(tdtype.name.lower())
            qmin, qmax = dtype2range(t_qinfo.dtype)
            q.bits = t_qinfo.bits
            # q.mins = [qmin]
            # q.maxs = [qmax]
            # q.mins = t_qinfo.mins.tolist()
            # q.maxs = t_qinfo.maxs.tolist()
            scheme = QScheme.str(t_qinfo.qscheme)
            q.scheme = Scheme.Asymmetric if 'asymmetric' in scheme else Scheme.Symmetric
            q.quantized_range = QuantizedRange.Restricted_Range if 'restricted_range' in scheme else QuantizedRange.Full_Range
            q.granularity = Granularity.Per_Channel if len(q.scales) > 1 else Granularity.Per_Tensor

            return q
        else:
            return None

    def get_use_input_QConfig(self):
        return self._use_input_QConfig

    def statistic(self, ori_tensor, qinfo):
        if isinstance(ori_tensor, (int, float)):
            qinfo.mins = ori_tensor if ori_tensor < 0 else 0
            qinfo.maxs = ori_tensor if ori_tensor > 0 else 0
            qinfo.scale = 1.0
            qinfo.zerop = 0
            qinfo.qinvariant = True
            qinfo.bits, qinfo.dtype = range2dtype(qinfo.mins, qinfo.maxs, True if qinfo.mins[0] < 0 else False)
            qinfo.qmin, qinfo.qmax = dtype2range(qinfo.dtype)
            return
        tensor = ori_tensor.detach().clone()
        cmode = qinfo.cmode
        qscheme = qinfo.qscheme
        s_min = qinfo.mins
        s_max = qinfo.maxs
        cur_min = None
        cur_max = None
        if 'per_tensor' in QScheme.str(qscheme):
            cur_min = tensor.min()
            cur_max = tensor.max()
        elif 'per_channel' in QScheme.str(qscheme):
            # now defaultly key_axis = 0
            if tensor.dim() > 0:
                cur_min = tensor.reshape([tensor.shape[0], -1]).min(dim=-1).values
                cur_max = tensor.reshape([tensor.shape[0], -1]).max(dim=-1).values
            else:
                cur_min = tensor.min()
                cur_max = tensor.max()
        else:
            QAT_ERROR(f"qscheme not in per-tensor or per-channel")

        if s_min is None or s_max is None:
            qinfo.mins = cur_min
            qinfo.maxs = cur_max
            return

        default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if CMode.str(cmode) == 'extrema':
            qinfo.mins = torch.min(cur_min.to(default_device), s_min.to(default_device))
            qinfo.maxs = torch.max(cur_max.to(default_device), s_max.to(default_device))
        elif CMode.str(cmode) == 'mean':
            qinfo.mins = qinfo.running_momentum * s_min + (1 - qinfo.running_momentum) * cur_min
            qinfo.maxs = qinfo.running_momentum * s_max + (1 - qinfo.running_momentum) * cur_max
        else:
            QAT_ERROR(f"cmode not in extrema and mean")

        qinfo.mins = torch.min(qinfo.mins.to(default_device), torch.tensor(0.0).to(default_device))
        qinfo.maxs = torch.max(qinfo.maxs.to(default_device), torch.tensor(0.0).to(default_device))

    def statistic_weight(self, weight_tensor):
        self.statistic(weight_tensor, self.weight_qinfo)

    def statistic_activation(self, activation_tensor):
        self.statistic(activation_tensor, self.activation_qinfo)

    @staticmethod
    def linear_affine(qinfo):
        QUANTIZE_ZERO_BAND = torch.finfo(torch.float32).eps

        f_min = qinfo.mins.detach().cpu().numpy()
        f_max = qinfo.maxs.detach().cpu().numpy()
        qbits = qinfo.bits
        dtype = qinfo.dtype
        qscheme = QScheme.str(qinfo.qscheme)

        q_min, q_max = bits2range(qbits, True)
        is_signed = True
        if dtype is not None and not is_float(dtype):
            # algo-determined: has deduce the dtype
            is_signed = dtype_is_signed(dtype)
            is_signed = (not np.all(f_min >= 0)) or is_signed
        elif dtype is None:
            # runtime-deteermined to deduce the dtype
            if np.all(f_min >= 0):
                is_signed = False
        elif is_float(dtype):
            return
        else:
            pass

        qinfo.dtype = bits2dtype(qbits, is_signed)
        q_max, q_min = 2 ** qbits - 1, 0
        if is_signed:
            if QuantMode.is_full_range(qscheme):
                q_max = 2 ** (qbits - 1) - 1
                q_min = -1 * q_max - 1
            else:
                q_max = 2 ** (qbits - 1) - 1
                q_min = -1 * q_max
        q_range = q_max - q_min

        if QuantMode.is_asymmetric(qscheme):
            f_ranges = f_max - f_min
            f_zfactor = f_min
            f_zoffset = q_min * np.ones_like(f_zfactor)
        else:
            f_ranges = np.maximum(np.abs(f_max), np.abs(f_min))
            if is_signed:
                f_ranges *= 2
            f_zfactor = np.zeros_like(f_min)
            f_zoffset = np.zeros_like(f_zfactor)

        q_ranges = q_range * np.ones_like(f_ranges)
        f_ranges = np.where(f_ranges < QUANTIZE_ZERO_BAND, q_ranges, f_ranges)
        f_ranges = np.where(np.isnan(f_ranges), q_ranges, f_ranges)
        scale = q_ranges / f_ranges
        zerop = np.clip(np.round(scale * f_zfactor) - f_zoffset, -2 ** (qbits - 1) + 1, 2 ** (qbits - 1))

        qinfo.scale = scale
        qinfo.zerop = zerop
        qinfo.qmin = q_min
        qinfo.qmax = q_max

        return scale, zerop, q_min, q_max

    def _clear_lower_bits_for_bias(self, tensor, qinfo):
        bias_effective_bits = qinfo.bias_effective_bits
        lmin, lmax = bits2range(bias_effective_bits, True)
        bmin = min(tensor.min().item(), -1)
        bmax = max(tensor.max().item(), 1)
        lbits = math.ceil(max(math.log2(bmax/lmax), math.log2(bmin/lmin)))
        if lbits > 0:
            tensor = torch.bitwise_left_shift(torch.bitwise_right_shift(tensor.long(), lbits), lbits).float()

        return tensor

    def _linear_quantize_dequantize(self, tensor, qinfo):
        if qinfo.qinvariant:
            return tensor
        scale, zerop, clip_min, clip_max = qinfo.scale, qinfo.zerop, qinfo.qmin, qinfo.qmax
        tscale = torch_tensor(scale, device=tensor.device)
        tzerop = torch_tensor(zerop, device=tensor.device)
        tscale = QInfo.broadcast(tscale, len(tensor.shape))
        tzerop = QInfo.broadcast(tzerop, len(tensor.shape))

        tensor = Round.apply(tensor * tscale - tzerop)
        tensor = Clamp.apply(tensor, clip_min, clip_max)
        if qinfo.bias_effective_bits != '' and isinstance(qinfo.bias_effective_bits, int) and qinfo.bias_effective_bits != qinfo.bits:
            QAT_DEBUG(f"now clear low bits")
            tensor = self._clear_lower_bits_for_bias(tensor, qinfo)
        tensor = (tensor + tzerop) / tscale
        return tensor

    def linear_quantize_dequantize(self, tensor, qinfo):
        if qinfo.qinvariant:
            return tensor
        _ = self.linear_affine(qinfo)
        return self._linear_quantize_dequantize(tensor, qinfo)

    def fake_quant(self, tensor, qinfo):
        if self._use_input_QConfig or self.quant_stage == QuantStage.FP32:
            return tensor
        elif self.quant_stage == QuantStage.CALIB:
            self.statistic(tensor, qinfo)
            _ = self.linear_affine(qinfo)
            return tensor
        elif self.quant_stage == QuantStage.QAT:
            self.statistic(tensor, qinfo)
            tensor = self.linear_quantize_dequantize(tensor, qinfo)
            return tensor
        else:  # self.quant_stage == QuantStage.INFER
            tensors = self._linear_quantize_dequantize(tensor, qinfo)
            # tensors = self.linear_quantize_dequantize(tensor, qinfo)
            return tensors

    def unify_shift(self, total_scale, bits):

        q = QInfo()
        q.qscheme = QScheme.str_to_qscheme('per_tensor_symmetric_full_range')
        q.cmode = CMode.str_to_cmode('extrema')
        q.bits = bits

        self.statistic(total_scale, q)
        ts_scale, ts_zp, ts_qmin, ts_qmax = self.linear_affine(q)
        round_func = torch.clone
        qtotal_scale = linear_quantize_clip(total_scale, ts_scale, ts_zp, ts_qmin, ts_qmax, round_func=round_func)
        ts_scale_doscale, _, ts_scale_doshift, _ = get_scale_approximation_params(1.0 / ts_scale, q.bits)
        # if ts_scale_doshift < 0:
        #     ts_scale_doscale *= 2 ** (0 - ts_scale_doshift)
        #     ts_scale_doshift = torch.zeros_like(ts_scale_doshift)

        # while ts_scale_doscale * 2 < 32767:
        #     ts_scale_doscale *= 2
        #     ts_scale_doshift += 1
        unify_ts = qtotal_scale * ts_scale_doscale.to(qtotal_scale.device)
        return unify_ts, ts_scale_doscale, ts_scale_doshift.to(qtotal_scale.device)

    def _refactor_weight_scale(self, wqinfo, input_scale, output_scale):
        weight_scale = wqinfo.scale
        total_scale = output_scale / input_scale / weight_scale

        unify_ts, doscale, doshift = self.unify_shift(total_scale, wqinfo.bits)

        unify_weight_scale = output_scale / input_scale / (unify_ts * 0.5 ** (doshift))
        print(f"original weight scale = {weight_scale}")
        print(f"unify weight scale = {unify_weight_scale}")

        wqinfo.scale = unify_weight_scale

    def _get_weight_scale(self, wqinfo, input_scale, output_scale):
        def _aiff_merge_shifts(scales, shifts):
            scales = scales.long()
            shifts = shifts.long()  # if shifts is 8bit, 2**shifts is still 8bit thus overflow
            m_shift = shifts.max()
            shifts = m_shift - shifts
            s = scales * (2**shifts)
            while s.max() > (1 << 15) - 1:
                s = s >> 1
                m_shift -= 1
            return s, m_shift
        lweight_scale = wqinfo.scale.detach().cpu().numpy()
        loutput_scale = output_scale.detach().cpu().numpy()
        linput_scale = input_scale.detach().cpu().numpy()
        total_scale = loutput_scale / linput_scale / lweight_scale
        doscale, _, doshift, _ = get_scale_approximation_params(torch.tensor(total_scale), wqinfo.bits)
        unify_scale, unify_shift = _aiff_merge_shifts(doscale, doshift)

        try:
            import copy
            new_total_scale = unify_scale * 0.5 ** unify_shift
            tmp = copy.deepcopy(new_total_scale)
            new_total_scale[new_total_scale == 0] = 1
            wnew_scale = loutput_scale / linput_scale / new_total_scale.detach().cpu().numpy()
            wnew_scale[tmp == 0] = 1.0
            if new_total_scale[new_total_scale == 0].numel() > 1:
                QAT_ERROR(f"should be non-zero of new_total_scale")
        except Exception as e:
            raise e
        t = wnew_scale
        a = np.argwhere(t == np.inf)
        wqinfo.scale = torch.tensor(t, device=wqinfo.scale.device)

        def scale_to_min_max(qinfo):
            qrange = qinfo.qmax - qinfo.qmin
            scale = qinfo.scale
            zp = qinfo.zerop
            fmax = (qrange + zp) / scale
            assert fmax.numel() == scale.numel(), f"failed"
            qinfo.maxs = fmax / 2
            qinfo.mins = -fmax / 2

        scale_to_min_max(wqinfo)

    def fake_quant_weight_with_unified(self, tensor, wqinfo, output_scale, input_scale):
        if self._use_input_QConfig or self.quant_stage == QuantStage.FP32:
            return tensor
        elif self.quant_stage == QuantStage.CALIB:
            self.statistic(tensor, wqinfo)
            return tensor
        elif self.quant_stage == QuantStage.QAT:
            self.statistic(tensor, wqinfo)
            self.linear_affine(wqinfo)
            # self._refactor_weight_scale(wqinfo, input_scale, output_scale)
            self._get_weight_scale(wqinfo, input_scale, output_scale)
            tensor = self._linear_quantize_dequantize(tensor, wqinfo)
            return tensor
        else:  # self.quant_stage == QuantStage.INFER
            tensors = self.linear_quantize_dequantize(tensor, wqinfo)
            return tensors
