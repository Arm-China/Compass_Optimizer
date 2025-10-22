# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import *
import math

# only for experiment, not a real Zhouyi NPU operator
register_optype('SDPAttention')


@quant_register(OpType.SDPAttention)
def SDPAttention_quant(self, *args):
    Q = self.inputs[0]
    K = self.inputs[1]
    V = self.inputs[2]
    out = self.outputs[0]
    dev = out.betensor.device
    q_broadcast_scale = Q.broadcast_scale.transpose(-1, -2) if self.get_param('trans_q') else Q.broadcast_scale
    k_broadcast_scale = K.broadcast_scale.transpose(-1, -2) if self.get_param('trans_k') else K.broadcast_scale
    v_broadcast_scale = V.broadcast_scale.transpose(-1, -2) if self.get_param('trans_v') else V.broadcast_scale
    min_compatible_zhouyi_target = self.attrs["min_compatible_zhouyi_target"].upper()
    ztarget = Target.optimized_target_level(min_compatible_zhouyi_target)
    q_mode_activation = QuantMode.to_symmetric(self.attrs["q_mode_activation"])
    q_bits_activation = self.attrs["q_bits_activation"]

    out.qbits = q_bits_activation
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = \
        get_linear_quant_params_from_tensor(out,
                                            q_mode_activation,
                                            out.qbits,
                                            is_signed=is_signed(V.dtype))
    out.qinvariant = False

    if ztarget < 2:
        self.params['impl_method'] = 'vanilla_attn'

        dqd_bits = 8
        qxk_bits = 12
        exp_val_bits = 16

        # for k -= k_mean
        smoothed_k = self.placeholders[0]
        smoothed_k.qbits = q_bits_activation
        smoothed_k.scale, smoothed_k.zerop, smoothed_k.qmin, smoothed_k.qmax, smoothed_k.dtype = \
            get_linear_quant_params_from_tensor(smoothed_k,
                                                q_mode_activation,
                                                smoothed_k.qbits,
                                                is_signed=True)
        smoothed_k.qinvariant = False

        def dqd_quant(dqd_betensor, dqdbits, tname):
            dqd = PyTensor('smoothed_k_dqd', dqd_betensor)
            dqd.qbits = dqdbits
            dqd.min, dqd.max = 0.0, dqd.betensor.max().item()
            dqd.scale, dqd.zerop, dqd.qmin, dqd.qmax, dqd.dtype = \
                get_linear_quant_params_from_tensor(dqd,
                                                    QuantMode.to_per_tensor(q_mode_activation),
                                                    dqd.qbits,
                                                    is_signed=False)
            dqd.qinvariant = False
            dqd.betensor = linear_quantize_clip(dqd.betensor, dqd.scale, dqd.zerop, dqd.qmin, dqd.qmax)
            self.constants[tname+'_dqd'] = dqd
            dqd_do_scale, dqd_do_scale_type, dqd_do_shift, dqd_do_shift_type = \
                get_scale_approximation_params(1.0 / dqd.scale,
                                               mult_bits=16,
                                               force_shift_positive=True)
            self.params[tname+'_scale_value'] = dqd_do_scale
            self.params[tname+'_scale_type'] = dqd_do_scale_type
            self.params[tname+'_shift_value'] = dqd_do_shift
            self.params[tname+'_shift_type'] = dqd_do_shift_type
        dqd_quant(smoothed_k.broadcast_scale / k_broadcast_scale, dqd_bits, 'smoothed_k')
        # for qxk
        qxk = self.placeholders[1]
        qxk.qbits = qxk_bits
        qxk.scale, qxk.zerop, qxk.qmin, qxk.qmax, qxk.dtype = \
            get_linear_quant_params_from_tensor(qxk,
                                                QuantMode.to_per_tensor(q_mode_activation),
                                                qxk.qbits,
                                                is_signed=True)
        self.params['qxk_bits'] = qxk.qbits
        qxk.qinvariant = False
        dqd_quant(self.get_param('scale_factor') * qxk.scale /
                  (q_broadcast_scale * smoothed_k.broadcast_scale), dqd_bits, 'qxk')
        # self.constants['qxk_m_i'] = PyTensor('qxk_m_i', linear_quantize_clip(qxk.max_key_axis, qxk.scale, qxk.zerop, 0, 2**(qxk.qbits-1)-1), dtype=bits2dtype(qxk_bits, is_signed=True))
        # for softmax
        max_val = torch.tensor((1 << exp_val_bits) - 1, device=dev)
        max_inp = torch.log(max_val)
        lsteps = 256
        quant_range_linspace = torch.linspace(0, 2 ** qxk_bits - 1, steps=lsteps, device=dev)
        max_inp = max_inp - qxk.zerop / qxk.scale
        lut = linear_dequantize(quant_range_linspace - (2 ** qxk_bits - 1), qxk.scale, qxk.zerop) + max_inp
        lut = torch.exp(lut).round().clamp(0, max_val)
        self.constants["lut"] = PyTensor(self.name + "/explut",
                                         lut.cpu().numpy().astype(dtype2nptype(range2dtype(0, max_val.item())[1])))
        # for pxv
        dqd_quant(out.broadcast_scale / v_broadcast_scale, dqd_bits, 'pxv')
    else:
        self.params['impl_method'] = 'sage_attn'
        self.params['sage_attn_block_m'] = 128
        self.params['sage_attn_block_n'] = 64

        # for k -= k_mean
        smoothed_k = self.placeholders[0]
        smoothed_k.qbits = 8
        smoothed_k.scale, smoothed_k.zerop, smoothed_k.qmin, smoothed_k.qmax, smoothed_k.dtype = \
            get_linear_quant_params_from_tensor(smoothed_k,
                                                q_mode_activation,
                                                smoothed_k.qbits,
                                                is_signed=True)
        smoothed_k.qinvariant = False
        self.constants['k_broadcast_scale'] = PyTensor('k_broadcast_scale', to_fp24(1.0 / k_broadcast_scale))
        self.constants['smoothed_k_broadcast_scale'] = PyTensor(
            'smoothed_k_broadcast_scale', to_fp24(smoothed_k.broadcast_scale))
        # for qxk
        self.constants['qxk_scale1'] = PyTensor('qxk_scale1', to_fp24(
            self.get_param('scale_factor') / q_broadcast_scale))
        self.constants['qxk_scale2'] = PyTensor('qxk_scale2', to_fp24(
            1.0 / k_broadcast_scale))  # smoothed_k.broadcast_scale))

        # for softmax
        lut_items_in_bits = Target.aiff_lut_items_in_bits(min_compatible_zhouyi_target)
        lut = 2 ** torch.linspace(0.0, 1.0, steps=2**lut_items_in_bits + 1, device=dev)
        lut = to_fp24(lut)
        self.constants["lut"] = PyTensor(self.name + "/fp24_lut", lut.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))

        # for pxv
        self.constants['pxv_scale1'] = PyTensor('pxv_scale1', to_fp24(1.0 / v_broadcast_scale))
        self.constants['pxv_scale2'] = PyTensor('pxv_scale2', to_fp24(out.broadcast_scale))


@approx_register(OpType.SDPAttention)
def SDPAttention_approx(self, *args):
    approx_params = self.get_attrs('approx_params', optional=True, default_value=[0])
    method = int(approx_params[0] if len(approx_params) > 0 else 0)
    min_compatible_zhouyi_target = self.attrs["min_compatible_zhouyi_target"].upper()
    lut_items_in_bits = Target.aiff_lut_items_in_bits(min_compatible_zhouyi_target)
    # if 1 == method and Target.optimized_target_level(min_compatible_zhouyi_target) >= 2:
    if True:
        inp = self.inputs[0]
        dev = inp.betensor.device
        lut = 2 ** torch.linspace(0.0, 1.0, steps=2**lut_items_in_bits + 1, device=dev)
        lut = to_fp24(lut)
        self.constants["lut"] = PyTensor(self.name + "/fp24_lut", lut.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
        self.params['is_perf_mode'] = True  # use fast approximate implementation of AIFF as much as possible
        self.params['lut_mode'] = 'EXP'

        # self.params['impl_method'] = 'vanilla_attn'
        # self.params['vanilla_attn_use_QMatMul_for_QK'] = True
        # self.params['vanilla_attn_scale_granularity_for_q'] = 0
        # self.params['vanilla_attn_scale_granularity_for_k'] = 0
        # self.params['vanilla_attn_use_QMatMul_for_PV'] = True
        # self.params['vanilla_attn_scale_granularity_for_v'] = 1

        self.params['impl_method'] = 'sage_attn'
        self.params['sage_attn_block_m'] = 64
        self.params['sage_attn_block_n'] = 32
        self.params['sage_attn_use_QMatMul_for_QK'] = True
        self.params['sage_attn_scale_granularity_for_q'] = 0
        self.params['sage_attn_scale_granularity_for_k'] = 0
        self.params['sage_attn_use_QMatMul_for_PV'] = True
        self.params['sage_attn_scale_granularity_for_v'] = 1
    else:
        # not suit for aiff, need use tpc to implement a higher accuracy version
        self.params['is_perf_mode'] = False


@op_register(OpType.SDPAttention)
def SDPAttention_forward(self, *args):
    Q = self.inputs[0]
    K = self.inputs[1]
    V = self.inputs[2]
    out = self.outputs[0]

    q = Q.betensor + Q.broadcast_zerop
    if self.get_param('trans_q'):
        q = q.transpose(-1, -2)
    k = K.betensor + K.broadcast_zerop
    if self.get_param('trans_k'):
        k = k.transpose(-1, -2)
    v = V.betensor + V.broadcast_zerop
    if self.get_param('trans_v'):
        v = v.transpose(-1, -2)
    if self.get_param('enable_gqa'):
        k = k.repeat_interleave(q.size(-3) // k.size(-3), -3)
        v = v.repeat_interleave(q.size(-3) // v.size(-3), -3)
    if self.quantized:
        impl_method = self.get_param('impl_method', optional=True, default_value='vanilla_attn').lower().strip()

        mask = 0.  # torch.zeros([Batches, Heads, M, N], dtype=torch.float32, device=q.device)
        if self.get_param('with_mask'):
            mask = self.inputs[3].betensor.int()

        if 'sage_attn' == impl_method:
            k_broadcast_scale = self.constants['k_broadcast_scale'].betensor
            smoothed_k_broadcast_scale = self.constants['smoothed_k_broadcast_scale'].betensor
            # k = (k + k.sum(dim=-1, keepdim=True) * (-1.0 /
            #      k.shape[-1])) * k_broadcast_scale * smoothed_k_broadcast_scale
            # k = torch.clamp(k.round(), -128, 127)

            if self.get_param('with_mask'):
                mask = (mask + self.inputs[3].broadcast_zerop) * to_fp24(1.0 / self.inputs[3].broadcast_scale)

            pow2_f_lut = self.constants["lut"].betensor.float()
            scale_factor = self.get_param('scale_factor')
            BLOCK_M = self.get_param('sage_attn_block_m')
            BLOCK_N = self.get_param('sage_attn_block_n')
            M = q.size(-2)
            N = v.size(-2)
            FChannels = q.size(-1)
            Heads = q.size(-3)
            Batches = q.size(-4)
            Tm = math.ceil(M * 1.0 / BLOCK_M)
            Tn = math.ceil(N * 1.0 / BLOCK_N)
            o = torch.zeros_like(q)
            for i in range(Tm):
                m_start_idx = i * BLOCK_M
                m_end_idx = min(m_start_idx + BLOCK_M, M)
                m_i = torch.zeros([Batches, Heads, m_end_idx-m_start_idx],
                                  dtype=torch.float32, device=q.device) - float("inf")
                l_i = torch.zeros([Batches, Heads, m_end_idx-m_start_idx], dtype=torch.float32, device=q.device)
                acc = torch.zeros([Batches, Heads, m_end_idx-m_start_idx, FChannels],
                                  dtype=torch.float32, device=q.device)
                # q dim: batches, heads, seq_len, feature_channels
                q_block = q[:, :, m_start_idx:m_end_idx, :]
                q_scale = self.constants['qxk_scale1'].betensor[:, :, m_start_idx:m_end_idx, :]
                for j in range(Tn):
                    # k dim: batches, heads, feature_channels, seq_len
                    n_start_idx = j * BLOCK_N
                    n_end_idx = min(n_start_idx + BLOCK_N, N)
                    k_block = k[:, :, :, n_start_idx:n_end_idx]
                    k_scale = self.constants['qxk_scale2'].betensor[:, :, :, n_start_idx:n_end_idx]
                    # v dim: batches, heads, seq_len, feature_channels
                    v_block = v[:, :, n_start_idx:n_end_idx, :]

                    qk_block = torch.matmul(q_block.float(), k_block.float()) * q_scale * k_scale
                    if self.get_param('with_mask'):
                        qk_block += mask[:, :, m_start_idx:m_end_idx, n_start_idx:n_end_idx]
                    m_ij = torch.maximum(m_i, torch.max(qk_block, dim=-1)[0])
                    qk_block = qk_block - m_ij.unsqueeze(-1)
                    p_block = x3_aiff_exp_approximation(qk_block, pow2_f_lut)
                    l_ij = torch.sum(p_block, -1)
                    alpha = x3_aiff_exp_approximation(m_i - m_ij, pow2_f_lut)
                    l_i = l_i * alpha + l_ij
                    p_scale = 255.0
                    p_block = torch.clamp((p_block * p_scale).round(), 0, 255)
                    pv_block = torch.matmul(p_block.float(), v_block.float()) * \
                        (1.0 / p_scale) * self.constants['pxv_scale1'].betensor
                    acc = acc * alpha.unsqueeze(-1) + pv_block
                    m_i = m_ij
                # o dim: batches, heads, seq_len, feature_channels
                o[:, :, m_start_idx:m_end_idx, :] = (acc / l_i.unsqueeze(-1)) * \
                    self.constants['pxv_scale2'].betensor
            out.betensor = torch.clamp(o.round(), out.qmin, out.qmax)
        else:
            smoothed_k_dqd = self.constants['smoothed_k_dqd']
            smoothed_k_dqd_do_scale = self.params['smoothed_k_scale_value']
            smoothed_k_dqd_do_shift = self.params['smoothed_k_shift_value']
            k_mean = torch.round(k.sum(dim=-1, keepdim=True) * 1.0 / k.shape[-1])
            k = linear_requantize((k - k_mean) * smoothed_k_dqd.betensor, smoothed_k_dqd_do_scale,
                                  smoothed_k_dqd_do_shift, 0, bits2range(K.qbits, True)[0], bits2range(K.qbits, True)[1])

            qxk_dqd = self.constants['qxk_dqd']
            qxk_dqd_do_scale = self.params['qxk_scale_value']
            qxk_dqd_do_shift = self.params['qxk_shift_value']
            lut = self.constants["lut"]
            pxv_dqd = self.constants['pxv_dqd']
            pxv_dqd_do_scale = self.params['pxv_scale_value']
            pxv_dqd_do_shift = self.params['pxv_shift_value']
            if 'vanilla_attn' == impl_method:
                qxk_bits = self.params['qxk_bits']
                qk_block = linear_requantize(torch.matmul(q.float(), k.float(
                )) * qxk_dqd.betensor, qxk_dqd_do_scale,  qxk_dqd_do_shift, 0, bits2range(qxk_bits, True)[0], bits2range(qxk_bits, True)[1])
                if self.get_param('with_mask'):
                    dbits = max(0, qxk_bits - self.inputs[3].qbits)
                    qk_block += mask << dbits
                m_i = torch.max(qk_block, dim=-1)[0]
                qk_block = torch.clamp(qk_block - m_i.unsqueeze(-1) + 2**qxk_bits - 1, 0, 2**qxk_bits - 1)
                qk_shape = qk_block.shape
                p_block = lookup_lut_powerof2(qk_block.reshape((-1,)),
                                              lut.betensor,
                                              qxk_bits,
                                              False,
                                              dtype2bits(lut.dtype),
                                              is_signed(lut.dtype)).reshape(qk_shape)
                l_i = torch.sum(p_block, -1)
                p = torch.clamp((256.0 * p_block / l_i.unsqueeze(-1)).int(), 0, 255)
                out.betensor = linear_requantize(torch.matmul(p.float(), v.float(
                )) * pxv_dqd.betensor, pxv_dqd_do_scale, pxv_dqd_do_shift+8, 0, out.qmin, out.qmax)
            else:
                BLOCK_M = 64
                BLOCK_N = 1280
                M = q.size(-2)
                N = v.size(-2)
                FChannels = q.size(-1)
                Heads = q.size(-3)
                Batches = q.size(-4)
                Tm = math.ceil(M * 1.0 / BLOCK_M)
                Tn = math.ceil(N * 1.0 / BLOCK_N)
                o = torch.zeros_like(q)
                # qxk_m_i_offline = self.constants['qxk_m_i'].betensor
                for i in range(Tm):
                    m_start_idx = i * BLOCK_M
                    m_end_idx = min(m_start_idx + BLOCK_M, M)
                    m_i = torch.zeros([Batches, Heads, m_end_idx-m_start_idx],
                                      dtype=torch.int32, device=q.device) + torch.iinfo(torch.int32).min
                    l_i = torch.zeros([Batches, Heads, m_end_idx-m_start_idx], dtype=torch.int32, device=q.device)
                    acc = torch.zeros([Batches, Heads, m_end_idx-m_start_idx, FChannels],
                                      dtype=torch.int32, device=q.device)
                    # q dim: batches, heads, seq_len, feature_channels
                    q_block = q[:, :, m_start_idx:m_end_idx, :]
                    for j in range(Tn):
                        # k dim: batches, heads, feature_channels, seq_len
                        n_start_idx = j * BLOCK_N
                        n_end_idx = min(n_start_idx + BLOCK_N, N)
                        k_block = k[:, :, :, n_start_idx:n_end_idx]
                        # v dim: batches, heads, seq_len, feature_channels
                        v_block = v[:, :, n_start_idx:n_end_idx, :]
                        qxk_bits = self.params['qxk_bits']
                        qk_block = linear_requantize(torch.matmul(q_block.float(), k_block.float(
                        )) * qxk_dqd.betensor[:, :, m_start_idx:m_end_idx, n_start_idx:n_end_idx], qxk_dqd_do_scale,  qxk_dqd_do_shift, 0, bits2range(qxk_bits, True)[0], bits2range(qxk_bits, True)[1])
                        if self.get_param('with_mask'):
                            dbits = max(0, qxk_bits - self.inputs[3].qbits)
                            qk_block += mask[:, :, m_start_idx:m_end_idx, n_start_idx:n_end_idx] << dbits
                        m_ij = torch.maximum(m_i, torch.max(qk_block, dim=-1)[0])
                        # m_ij = qxk_m_i_offline[m_start_idx:m_end_idx].unsqueeze(0).unsqueeze(0)
                        qk_block = torch.clamp(qk_block - m_ij.unsqueeze(-1) + 2**qxk_bits - 1, 0, 2**qxk_bits - 1)
                        qk_shape = qk_block.shape
                        p_block = lookup_lut_powerof2(qk_block.reshape((-1,)),
                                                      lut.betensor,
                                                      qxk_bits,
                                                      False,
                                                      dtype2bits(lut.dtype),
                                                      is_signed(lut.dtype)).reshape(qk_shape)
                        l_ij = torch.sum(p_block, -1)
                        alpha = lookup_lut_powerof2(torch.clamp((m_i - m_ij + + 2**qxk_bits - 1).reshape((-1,)), 0, 2**qxk_bits - 1),
                                                    lut.betensor,
                                                    qxk_bits,
                                                    False,
                                                    dtype2bits(lut.dtype),
                                                    is_signed(lut.dtype)).reshape(m_i.shape)
                        # alpha = 1.0
                        l_i = l_i * alpha + l_ij
                        pv_block = linear_requantize(torch.matmul(p_block.float(), v_block.float(
                        )) * pxv_dqd.betensor, pxv_dqd_do_scale, pxv_dqd_do_shift, 0, torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max)
                        acc = acc * alpha.unsqueeze(-1) + pv_block
                        # acc = acc * alpha + pv_block
                        m_i = m_ij
                    # o dim: batches, heads, seq_len, feature_channels
                    o[:, :, m_start_idx:m_end_idx, :] = (acc * 1.0 / l_i.unsqueeze(-1)).int()
                out.betensor = o
    else:
        # subtract mean value on seqlen dim for smoothing K
        k -= k.mean(dim=-1, keepdim=True)

        impl_method = self.get_param('impl_method', optional=True, default_value='vanilla_attn').lower().strip()
        if self.approximated and 'vanilla_attn' == impl_method:
            pow2_f_lut = self.constants["lut"].betensor.float()
            scale_factor = self.get_param('scale_factor')
            vanilla_eps = torch.finfo(torch.float32).eps
            vanilla_qmin, vanilla_qmax = -128.0, 127.0
            q_block = q * scale_factor
            q_scale = 1.0
            k_block = k
            k_scale = 1.0
            v_block = v
            v_scale = 1.0
            if self.get_param('vanilla_attn_use_QMatMul_for_QK', optional=True, default_value=True):
                # online quantize Q, K
                max_over_c, _ = torch.max(torch.abs(q_block), dim=-1, keepdim=True)
                max_over_sc, _ = torch.max(max_over_c, dim=-2, keepdim=True)
                q_scale = (max_over_c if self.get_param('vanilla_attn_scale_granularity_for_q')
                           == 1 else max_over_sc) / vanilla_qmax
                q_block = torch.clamp((q_block / (q_scale + vanilla_eps)).round(), vanilla_qmin, vanilla_qmax)
                max_over_c, _ = torch.max(torch.abs(k_block), dim=-2, keepdim=True)
                max_over_sc, _ = torch.max(max_over_c, dim=-1, keepdim=True)
                k_scale = (max_over_c if self.get_param('vanilla_attn_scale_granularity_for_k')
                           == 1 else max_over_sc) / vanilla_qmax
                k_block = torch.clamp((k_block / (k_scale + vanilla_eps)).round(), vanilla_qmin, vanilla_qmax)
                max_over_c, _ = torch.max(torch.abs(v_block), dim=-2, keepdim=True)
                max_over_sc, _ = torch.max(max_over_c, dim=-1, keepdim=True)
                v_scale = (max_over_c if self.get_param('vanilla_attn_scale_granularity_for_v')
                           == 1 else max_over_sc) / vanilla_qmax
                v_block = torch.clamp((v_block / (v_scale + vanilla_eps)).round(), vanilla_qmin, vanilla_qmax)
            qk_block = torch.matmul(q_block.float(), k_block.float()) * q_scale * k_scale
            if self.get_param('with_mask'):
                mask = self.inputs[3].betensor
                qk_block += mask
            p_block = x3_aiff_softmax_approximation(qk_block, -1, pow2_f_lut)
            p_scale = 1.0
            if self.get_param('vanilla_attn_use_QMatMul_for_PV', optional=True, default_value=True):
                # online quantize P, V
                # always use per-tensor scale for P, as 0 <= p <= 1
                p_scale = vanilla_qmax - vanilla_qmin
                p_block = torch.clamp((p_block * p_scale).round(), 0, vanilla_qmax - vanilla_qmin)
            pv_block = torch.matmul(p_block.float(), v_block.float()) * (v_scale / p_scale)
            out.betensor = pv_block
        elif self.approximated and 'sage_attn' == impl_method:
            pow2_f_lut = self.constants["lut"].betensor.float()
            scale_factor = self.get_param('scale_factor')
            BLOCK_M = self.get_param('sage_attn_block_m')
            BLOCK_N = self.get_param('sage_attn_block_n')
            M = q.size(-2)
            N = v.size(-2)
            FChannels = q.size(-1)
            Heads = q.size(-3)
            Batches = q.size(-4)
            Tm = math.ceil(M * 1.0 / BLOCK_M)
            Tn = math.ceil(N * 1.0 / BLOCK_N)
            k_block_dict = {}
            k_scale_dict = {}
            v_block_dict = {}
            v_scale_dict = {}
            sage_eps = torch.finfo(torch.float32).eps
            sage_qmin, sage_qmax = -128.0, 127.0
            mask = 0.  # torch.zeros([Batches, Heads, M, N], dtype=torch.float32, device=q.device)
            if self.get_param('with_mask'):
                mask = self.inputs[3].betensor
            o = torch.zeros_like(q)
            for i in range(Tm):
                m_start_idx = i * BLOCK_M
                m_end_idx = min(m_start_idx + BLOCK_M, M)
                m_i = torch.zeros([Batches, Heads, m_end_idx-m_start_idx],
                                  dtype=torch.float32, device=q.device) - float("inf")
                l_i = torch.ones([Batches, Heads, m_end_idx-m_start_idx], dtype=torch.float32, device=q.device)
                acc = torch.zeros([Batches, Heads, m_end_idx-m_start_idx, FChannels],
                                  dtype=torch.float32, device=q.device)
                # q dim: batches, heads, seq_len, feature_channels
                q_block = q[:, :, m_start_idx:m_end_idx, :] * scale_factor
                q_scale = 1.0
                if self.get_param('sage_attn_use_QMatMul_for_QK', optional=True, default_value=True):
                    # online quantize Q, K
                    max_over_c, _ = torch.max(torch.abs(q_block), dim=-1, keepdim=True)
                    max_over_sc, _ = torch.max(max_over_c, dim=-2, keepdim=True)
                    q_scale = (max_over_c if self.get_param('sage_attn_scale_granularity_for_q')
                               == 1 else max_over_sc) / sage_qmax
                    q_block = torch.clamp((q_block / (q_scale + sage_eps)).round(), sage_qmin, sage_qmax)
                for j in range(Tn):
                    if 0 == i:
                        # k dim: batches, heads, feature_channels, seq_len
                        n_start_idx = j * BLOCK_N
                        n_end_idx = min(n_start_idx + BLOCK_N, N)
                        k_block = k[:, :, :, n_start_idx:n_end_idx]
                        k_scale = 1.0
                        if self.get_param('sage_attn_use_QMatMul_for_QK', optional=True, default_value=True):
                            # online quantize Q, K
                            max_over_c, _ = torch.max(torch.abs(k_block), dim=-2, keepdim=True)
                            max_over_sc, _ = torch.max(max_over_c, dim=-1, keepdim=True)
                            k_scale = (max_over_c if self.get_param('sage_attn_scale_granularity_for_k')
                                       == 1 else max_over_sc) / sage_qmax
                            k_block = torch.clamp((k_block / (k_scale + sage_eps)).round(), sage_qmin, sage_qmax)
                        k_block_dict[j] = k_block
                        k_scale_dict[j] = k_scale
                        # v dim: batches, heads, seq_len, feature_channels
                        v_block = v[:, :, n_start_idx:n_end_idx, :]
                        v_scale = 1.0
                        if self.get_param('sage_attn_use_QMatMul_for_PV', optional=True, default_value=True):
                            # online quantize P, V
                            max_over_c, _ = torch.max(torch.abs(v_block), dim=-2, keepdim=True)
                            max_over_sc, _ = torch.max(max_over_c, dim=-1, keepdim=True)
                            v_scale = (max_over_c if self.get_param('sage_attn_scale_granularity_for_v')
                                       == 1 else max_over_sc) / sage_qmax
                            v_block = torch.clamp((v_block / (v_scale + sage_eps)).round(), sage_qmin, sage_qmax)
                        v_block_dict[j] = v_block
                        v_scale_dict[j] = v_scale
                    k_block = k_block_dict[j]
                    k_scale = k_scale_dict[j]
                    v_block = v_block_dict[j]
                    v_scale = v_scale_dict[j]
                    qk_block = torch.matmul(q_block.float(), k_block.float()) * q_scale * k_scale
                    if self.get_param('with_mask'):
                        qk_block += mask[:, :, m_start_idx:m_end_idx, n_start_idx:n_end_idx]
                    m_ij = torch.maximum(m_i, torch.max(qk_block, dim=-1)[0])
                    qk_block = qk_block - m_ij.unsqueeze(-1)
                    p_block = x3_aiff_exp_approximation(qk_block, pow2_f_lut)
                    l_ij = torch.sum(p_block, -1)
                    alpha = x3_aiff_exp_approximation(m_i - m_ij, pow2_f_lut)
                    l_i = l_i * alpha + l_ij
                    p_scale = 1.0
                    if self.get_param('sage_attn_use_QMatMul_for_PV', optional=True, default_value=True):
                        # online quantize P, V
                        # always use per-tensor scale for P, as 0 <= p <= 1
                        p_scale = sage_qmax - sage_qmin
                        p_block = torch.clamp((p_block * p_scale).round(), 0, sage_qmax - sage_qmin)
                    pv_block = torch.matmul(p_block.float(), v_block.float()) * (v_scale / p_scale)
                    acc = acc * alpha.unsqueeze(-1) + pv_block
                    m_i = m_ij
                # o dim: batches, heads, seq_len, feature_channels
                o[:, :, m_start_idx:m_end_idx, :] = acc / l_i.unsqueeze(-1)
            out.betensor = o
        else:
            if len(self.placeholders) < 1:
                ph0 = PyTensor(self.name+"/smoothed_k", k.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
                ph0.key_axis = len(k.shape) - 1
                self.placeholders.append(ph0)
            self.placeholders[0].betensor = k

            qk = torch.matmul(q, k)
            scale_factor = self.get_param('scale_factor')
            qk = qk * scale_factor

            if len(self.placeholders) < 2:
                ph1 = PyTensor(self.name+"/qxk_normed", qk.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
                self.placeholders.append(ph1)
                # ph1.key_axis = len(qk.shape) - 2
            self.placeholders[1].betensor = qk

            if self.get_param('with_mask'):
                mask = self.inputs[3].betensor
                qk = qk + mask
            attn_weight = torch.softmax(qk, dim=-1)
            # ##################################################################################
            # ##online softmax
            # lut = 2 ** torch.linspace(0.0, 1.0, steps=2**9 + 1, device=qk.device)
            # pow2_f_lut = to_fp24(lut)
            # BLOCK_M = 64
            # BLOCK_N = 32
            # M = q.size(-2)
            # N = v.size(-2)
            # FChannels = q.size(-1)
            # Heads = q.size(-3)
            # Batches = q.size(-4)
            # Tm = math.ceil(M * 1.0 / BLOCK_M)
            # Tn = math.ceil(N * 1.0 / BLOCK_N)
            # attn_weight_o = torch.zeros_like(qk)
            # for i in range(Tm):
            #     m_start_idx = i * BLOCK_M
            #     m_end_idx = min(m_start_idx + BLOCK_M, M)
            #     m_i = torch.zeros([Batches, Heads, m_end_idx-m_start_idx],
            #                       dtype=torch.float32, device=q.device) - float("inf")
            #     l_i = torch.ones([Batches, Heads, m_end_idx-m_start_idx], dtype=torch.float32, device=q.device)
            #     p = torch.zeros([Batches, Heads, m_end_idx-m_start_idx, N],
            #                       dtype=torch.float32, device=q.device)
            #     alpha_record = torch.ones([Batches, Heads, m_end_idx-m_start_idx, Tn+1],
            #                       dtype=torch.float32, device=q.device)
            #     # qk dim: batches, heads, src_seq_len, tgt_seq_len
            #     for j in range(Tn):
            #         n_start_idx = j * BLOCK_N
            #         n_end_idx = min(n_start_idx + BLOCK_N, N)
            #         qk_block = qk[:,:,m_start_idx:m_end_idx,n_start_idx:n_end_idx]

            #         m_ij = torch.maximum(m_i, torch.max(qk_block, dim=-1)[0])
            #         qk_block = qk_block - m_ij.unsqueeze(-1)
            #         p_block = x3_aiff_exp_approximation(qk_block, pow2_f_lut) #torch.exp(qk_block) #
            #         l_ij = torch.sum(p_block, -1)
            #         alpha = x3_aiff_exp_approximation(m_i - m_ij, pow2_f_lut) #torch.exp(m_i - m_ij) #
            #         l_i = l_i * alpha + l_ij
            #         p[:,:,:,n_start_idx:n_end_idx] = p_block
            #         m_i = m_ij
            #         alpha_record[:,:,:,j] = alpha
            #     alpha_prod = torch.ones_like(alpha_record)
            #     for j in range(Tn-1,0,-1):
            #         alpha_prod[:,:,:,j] = alpha_prod[:,:,:,j+1] * alpha_record[:,:,:,j]
            #     for j in range(Tn):
            #         n_start_idx = j * BLOCK_N
            #         n_end_idx = min(n_start_idx + BLOCK_N, N)
            #         p[:,:,:,n_start_idx:n_end_idx] *= alpha_prod[:,:,:,j+1].unsqueeze(-1)
            #     attn_weight_o[:, :, m_start_idx:m_end_idx, :] = p / l_i.unsqueeze(-1)
            # attn_diff = attn_weight_o - attn_weight
            # OPT_INFO(f'online softmax vs naive softmax: {attn_diff.abs().max()}')
            # ##################################################################################
            out.betensor = torch.matmul(attn_weight.to(v.dtype), v)
    return out.betensor
