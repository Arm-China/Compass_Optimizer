# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.ops.rnn import *
from AIPUBuilder.Optimizer.logger import *
import torch.nn as nn


split_weights_name = ['wx_gk', 'wh_gk', 'wx_ck', 'wh_ck']


def list_equal(left, right):
    if isinstance(left, list) and isinstance(right, list):
        return left == right
    return False


@op_register(OpType.GRUv3)
def gruv3(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]

    current_batch_idx = self.current_batch_idx
    input_seq = inp0.betensor.float()
    initial_batch = self.inputs[1].betensor.shape[0]
    batch_size = inp0.betensor.shape[0]
    time_step = self.get_param('time_steps')
    input_size = self.get_param('input_size')
    cell_size = self.get_param('cell_size')
    direction = self.get_param('direction')
    threshold = self.get_param('threshold', optional=True, default_value=float('inf'))

    if direction == 'reverse':
        input_seq = torch.flip(input_seq, [1])

    initial_H = inp1.betensor.float()
    start_data_idx = self.current_batch_idx * batch_size

    start_initial_idx = start_data_idx % initial_batch
    in_state = initial_H[start_initial_idx: start_initial_idx + 1]
    for initial_idx in range(1, batch_size):
        current_initial_idx = (start_data_idx + initial_idx) % initial_batch
        in_state = torch.cat((in_state, initial_H[current_initial_idx: current_initial_idx + 1]), dim=0)

    w = self.constants["weights"].betensor.float()
    bias = self.constants['biases'].betensor.float()
    out_sequence = self.get_param('out_sequence')
    activations_list = self.get_param('activations') \
        if 'activations' in self.params else ['SIGMOID', 'TANH']

    state_batch = torch.zeros([batch_size, time_step, cell_size], device=inp0.betensor.device)
    state_last = torch.zeros([batch_size, cell_size], device=inp0.betensor.device)

    weights = w.permute(1, 0).float()
    gates_kernel = weights[:, :2 * cell_size]
    candidate_kernel = weights[:, 2 * cell_size:]

    if not self.quantized:
        # save weights to placeholder for caculating max_key_axis and min_key_axis through per-channel
        f_lut_in = torch.zeros([batch_size, time_step, 2*cell_size], device=inp0.betensor.device)
        f_lut_out = torch.zeros([batch_size, time_step, 2*cell_size], device=inp0.betensor.device)
        g_lut_in = torch.zeros([batch_size, time_step, cell_size], device=inp0.betensor.device)
        g_lut_out = torch.zeros([batch_size, time_step, cell_size], device=inp0.betensor.device)
        hidden_gate_out = torch.zeros([batch_size, time_step, cell_size], device=inp0.betensor.device)

        wx_gk = gates_kernel[0:input_size, :].permute(1, 0)
        wh_gk = gates_kernel[input_size:, :].permute(1, 0)
        wx_ck = candidate_kernel[0:input_size, :].permute(1, 0)
        wh_ck = candidate_kernel[input_size:, :].permute(1, 0)
        for idx, weights_name in enumerate(split_weights_name):
            if weights_name not in self.constants:
                split_weights = eval(weights_name)
                self.constants[weights_name] = PyTensor(
                    self.name+"/constants"+str(idx), split_weights.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
                self.constants[weights_name].betensor = split_weights
                self.constants[weights_name].ir_shape = TensorShape(list(split_weights.cpu().numpy().shape))
                self.constants[weights_name].ir_dtype = self.constants[weights_name].dtype

        gates_bias = bias[: 2 * cell_size]
        candidate_bias = bias[2 * cell_size: 3 * cell_size]
        for b in range(batch_size):
            state = torch.unsqueeze(in_state[b], dim=0)
            state_all = torch.zeros([1, cell_size], device=inp0.betensor.device, dtype=torch.float)
            for ts in range(time_step):
                in_ts = torch.reshape(input_seq[b, ts, :], (-1, input_size))
                if 'version' in self.params and self.params['version'] == "GRUV1":

                    gate_kernel_x_f = gates_kernel[:input_size, :]
                    gate_kernel_h_f = gates_kernel[input_size:, :]
                    candidate_kernel_x_f = candidate_kernel[:input_size, :]
                    candidate_kernel_h_f = candidate_kernel[input_size:, :]

                    hidden_bias = bias[3 * cell_size:]
                    h_by_wg_f = torch.matmul(state, gate_kernel_h_f)
                    x_by_wg_f = torch.add(torch.matmul(in_ts, gate_kernel_x_f),
                                          torch.unsqueeze(gates_bias, 0))

                    mat_sum_f = x_by_wg_f + h_by_wg_f
                    mat_sum_f = torch.clamp(mat_sum_f, -threshold, threshold)
                    f_lut_in[b, ts, :] = torch.squeeze(mat_sum_f, 0)
                    sig_lut_out_f = g_rnn_activation_func[activations_list[0]][1](mat_sum_f)
                    f_lut_out[b, ts, :] = torch.squeeze(sig_lut_out_f, 0)
                    r_f, u_f = torch.chunk(sig_lut_out_f, 2, dim=1)

                    h_by_wc_f = torch.matmul(state, candidate_kernel_h_f)
                    h_by_wc_f = torch.add(h_by_wc_f, torch.unsqueeze(hidden_bias, 0))
                    hidden_gate_out[b, ts, :] = torch.squeeze(h_by_wc_f)
                    r_hwc_f = torch.multiply(r_f, h_by_wc_f)
                    x_by_wc_f = torch.add(torch.matmul(in_ts, candidate_kernel_x_f), torch.unsqueeze(candidate_bias, 0))
                    mat_sum2_f = x_by_wc_f + r_hwc_f
                    mat_sum2_f = torch.clamp(mat_sum2_f, -threshold, threshold)
                    g_lut_in[b, ts, :] = torch.squeeze(mat_sum2_f, 0)
                    c_f = g_rnn_activation_func[activations_list[1]][1](mat_sum2_f)
                    g_lut_out[b, ts, :] = torch.squeeze(c_f, 0)
                    state = torch.multiply((1.0 - u_f), c_f) + torch.multiply(u_f, state)
                else:  # GRUV3
                    gate_input = torch.cat((in_ts, state), dim=1)
                    sum0 = torch.add(torch.matmul(gate_input, gates_kernel), torch.unsqueeze(gates_bias, 0))  # [1,1024]
                    sum0 = torch.clamp(sum0, -threshold, threshold)
                    f_lut_in[b, ts, :] = torch.squeeze(sum0, 0)
                    f_out = g_rnn_activation_func[activations_list[0]][1](sum0)
                    f_lut_out[b, ts, :] = torch.squeeze(f_out, 0)
                    r, u = torch.chunk(f_out, 2, dim=1)
                    state_r = torch.multiply(state, r)

                    candidate_input = torch.cat((in_ts, state_r), dim=1)
                    sum1 = torch.add(torch.matmul(candidate_input, candidate_kernel),
                                     torch.unsqueeze(candidate_bias, 0))
                    sum1 = torch.clamp(sum1, -threshold, threshold)
                    g_lut_in[b, ts, :] = torch.squeeze(sum1, 0)
                    c = g_rnn_activation_func[activations_list[1]][1](sum1)
                    g_lut_out[b, ts, :] = torch.squeeze(c, 0)
                    state = torch.multiply((1.0 - u), c) + torch.multiply(u, state)
                state_all = state if ts == 0 else torch.cat((state_all, state), dim=0)

            state_last = state if b == 0 else torch.cat((state_last, state), dim=0)
            state_all = torch.unsqueeze(state_all, dim=0)
            state_batch = state_all if b == 0 else torch.cat((state_batch, state_all), dim=0)
        placeholders_list = ['state_batch', 'f_lut_in', 'f_lut_out', 'g_lut_in', 'g_lut_out', 'hidden_gate_out']
        if len(self.placeholders) < len(placeholders_list):
            for placeholder_name in placeholders_list:
                ph = PyTensor(self.name + '/' + placeholder_name,
                              eval(placeholder_name).cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
                self.placeholders.append(ph)
        for idx, placeholder_name in enumerate(placeholders_list):
            self.placeholders[idx].betensor = eval(placeholder_name)

    else:
        dtype = dtype2str(self.outputs[0].dtype)
        q_bits_activation = dtype2bits(self.outputs[0].dtype)
        if dtype not in ['int8', 'int16']:
            OPT_FATAL("Currently gruv3/gruv1 only  support quantization bits  of activations is 8 or 16")
        qmin, qmax = bits2range(q_bits_activation, True)

        lut_in_bits = self.inputs[0].qbits
        lut_out_bits = self.outputs[0].qbits

        biases = torch.squeeze(bias)
        gates_bias_q = biases[: 2 * cell_size]
        candidate_bias_q = biases[2 * cell_size:3 * cell_size]
        if 'version' in self.params and self.params['version'] == "GRUV1":
            hidden_bias_q = biases[3 * cell_size:]

        lut_names = ['lut_rt', 'lut_zt', 'lut_ht']
        rt_table = self.constants['lut_rt'].betensor
        zt_table = self.constants['lut_zt'].betensor
        ht_table = self.constants['lut_ht'].betensor

        wx_gk_q = gates_kernel[0:input_size, :]
        wh_gk_q = gates_kernel[input_size:, :]
        wx_ck_q = candidate_kernel[0:input_size, :]
        wh_ck_q = candidate_kernel[input_size:, :]
        if 'scale_value' in self.params:
            scale = torch.tensor(self.params["scale_value"], device=inp0.betensor.device)
            shift = torch.tensor(self.params["shift_value"], device=inp0.betensor.device)
        elif "scale" in self.constants:
            requant_scale = self.constants["scale"].betensor
            requant_shift = self.constants["shift"].betensor
            gk_step = wx_gk_q.shape[1]
            ck_step = wx_ck_q.shape[1]
            step_length = [gk_step, gk_step, ck_step, ck_step, 1, 1]
            start_idx = 0
            scale = []
            shift = []
            for idx, step in enumerate(step_length):
                scale.append(requant_scale[start_idx: start_idx + step].to(inp0.betensor.device))
                shift.append(requant_shift[start_idx: start_idx + step].to(inp0.betensor.device))
                start_idx += step

        # Here lib use the default value 15, but the default value of 15 may cause a loss of accuracy
        # and it is better that lib can use remain_shift
        aasrb = self.get_param('remain_shift', optional=True, default_value=10)
        act_qmin, act_qmax = -2 ** 31, 2 ** 31 - 1
        input_bits = dtype2bits(self.inputs[0].dtype)
        shift0_zeros_tensor = torch.zeros_like(shift[0], device=inp0.betensor.device)
        shift1_zeros_tensor = torch.zeros_like(shift[1], device=inp0.betensor.device)
        shift2_zeros_tensor = torch.zeros_like(shift[2], device=inp0.betensor.device)
        shift3_zeros_tensor = torch.zeros_like(shift[3], device=inp0.betensor.device)
        mtp_trsh_rz_h = shift0_zeros_tensor if input_bits <= 8 else torch.max(shift[0] - aasrb, shift0_zeros_tensor)
        mtp_trsh_rz_x = shift1_zeros_tensor if input_bits <= 8 else torch.max(shift[1] - aasrb, shift1_zeros_tensor)
        itp_trsh_rz_h = shift0_zeros_tensor if input_bits <= 8 else torch.min(
            shift0_zeros_tensor+aasrb, shift[0]) + mtp_trsh_rz_x
        itp_trsh_rz = shift1_zeros_tensor if input_bits <= 8 else torch.min(shift1_zeros_tensor+aasrb, shift[1])

        mtp_trsh_c_h = shift2_zeros_tensor if input_bits <= 8 else torch.max(shift[2] - aasrb, shift2_zeros_tensor)
        mtp_trsh_c_x = shift3_zeros_tensor if input_bits <= 8 else torch.max(shift[3] - aasrb, shift3_zeros_tensor)
        itp_trsh_c_h = shift2_zeros_tensor if input_bits <= 8 else torch.min(
            shift2_zeros_tensor+aasrb, shift[2]) + mtp_trsh_c_x
        itp_trsh_c = shift3_zeros_tensor if input_bits <= 8 else torch.min(shift3_zeros_tensor+aasrb, shift[3])

        for b in range(batch_size):
            state = torch.unsqueeze(in_state[b], dim=0).float()
            state_all = torch.zeros([1, cell_size], device=inp0.betensor.device, dtype=torch.float)
            for ts in range(time_step):
                in_ts = torch.reshape(input_seq[b, ts, :], (-1, input_size)).float()
                x_by_wg = torch.matmul(in_ts, wx_gk_q)
                h_by_wg = torch.matmul(state, wh_gk_q)
                if input_bits <= 8:
                    re_scaled_h_by_wg = linear_requantize(h_by_wg, scale[0], shift[0], 0, act_qmin, act_qmax)

                    mat_sum = x_by_wg + re_scaled_h_by_wg + torch.unsqueeze(gates_bias_q, 0)
                    rescaled_mat_sum = linear_requantize(mat_sum, scale[1], shift[1], 0, qmin, qmax)
                    rt_zt_lut_out = lookup_lut_powerof2(rescaled_mat_sum, rt_table,
                                                        lut_in_bits, True, lut_out_bits, True)
                    r, u = torch.chunk(rt_zt_lut_out, 2, dim=1)

                    x_by_wc = torch.matmul(in_ts, wx_ck_q)
                    if 'version' in self.params and self.params['version'] == "GRUV1":
                        hidden_scale = self.params["hidden_scale_value"]
                        hidden_shift = self.params["hidden_shift_value"]
                        h_by_wc = torch.add(torch.matmul(state, wh_ck_q), torch.unsqueeze(hidden_bias_q, 0))
                        h_by_wc = linear_requantize(h_by_wc, hidden_scale, hidden_shift, 0, qmin, qmax)
                        h_by_wc = torch.multiply(r, h_by_wc)
                    else:
                        factor = 256.0
                        hprev_r = torch.round(torch.div(torch.multiply(r, state), factor))
                        h_by_wc = torch.matmul(hprev_r, wh_ck_q)
                    re_scaled_h_by_wc = linear_requantize(h_by_wc, scale[2], shift[2], 0, act_qmin, act_qmax)
                    mat_sum2 = x_by_wc + re_scaled_h_by_wc + torch.unsqueeze(candidate_bias_q, 0)

                    rescaled_mat_sum2 = linear_requantize(mat_sum2, scale[3], shift[3], 0, qmin, qmax)
                else:  # int16
                    h_by_wg = linear_requantize(h_by_wg, 1, mtp_trsh_rz_h, 0, act_qmin, act_qmax)
                    re_scaled_h_by_wg = linear_requantize(h_by_wg, scale[0], itp_trsh_rz_h, 0, act_qmin, act_qmax)
                    x_by_wg = linear_requantize(x_by_wg, 1, mtp_trsh_rz_x, 0, act_qmin, act_qmax)
                    gates_bias_q = linear_requantize(gates_bias_q, 1, mtp_trsh_rz_x, 0, act_qmin, act_qmax)
                    mat_sum = x_by_wg + re_scaled_h_by_wg + torch.unsqueeze(gates_bias_q, 0)
                    rescaled_mat_sum = linear_requantize(mat_sum, scale[1], itp_trsh_rz, 0, qmin, qmax)
                    rt_zt_lut_out = lookup_lut_powerof2(rescaled_mat_sum, rt_table,
                                                        lut_in_bits, True, lut_out_bits, True)
                    r, u = torch.chunk(rt_zt_lut_out, 2, dim=1)

                    x_by_wc = torch.matmul(in_ts, wx_ck_q)
                    if 'version' in self.params and self.params['version'] == "GRUV1":
                        hidden_scale = self.params["hidden_scale_value"]
                        hidden_shift = self.params["hidden_shift_value"]
                        hidden_bias_q = linear_requantize(hidden_bias_q, 1, mtp_trsh_c_h, 0, act_qmin, act_qmax)
                        h_by_wc = torch.add(torch.matmul(state, wh_ck_q), torch.unsqueeze(hidden_bias_q, 0))
                        h_by_wc = linear_requantize(h_by_wc, hidden_scale, hidden_shift, 0, qmin, qmax)
                        h_by_wc = torch.multiply(r, h_by_wc)
                    else:
                        factor = 65536.0
                        hprev_r = torch.round(torch.div(torch.multiply(r, state), factor))
                        h_by_wc = torch.matmul(hprev_r, wh_ck_q)
                    x_by_wc = linear_requantize(x_by_wc, 1, mtp_trsh_c_x, 0, act_qmin, act_qmax)
                    h_by_wc = linear_requantize(h_by_wc, 1, mtp_trsh_c_h, 0, act_qmin, act_qmax)
                    re_scaled_h_by_wc = linear_requantize(h_by_wc, scale[2], itp_trsh_c_h, 0, act_qmin, act_qmax)
                    candidate_bias_q = linear_requantize(candidate_bias_q, 1, mtp_trsh_c_x, 0, act_qmin, act_qmax)
                    mat_sum2 = x_by_wc + re_scaled_h_by_wc + torch.unsqueeze(candidate_bias_q, 0)
                    rescaled_mat_sum2 = linear_requantize(mat_sum2, scale[3], itp_trsh_c, 0, qmin, qmax)
                c = lookup_lut_powerof2(rescaled_mat_sum2, ht_table, lut_in_bits, True, lut_out_bits, True)
                max_pre = qmax
                c_times_1_minus_u = (max_pre - u) * c
                rescaled_c_times_1_minus_u = linear_requantize(
                    c_times_1_minus_u, scale[4], shift[4], 0, act_qmin, act_qmax)
                state = rescaled_c_times_1_minus_u + u * state
                state = linear_requantize(state, scale[5], shift[5], 0, qmin+1, qmax)
                state_all = state if ts == 0 else torch.cat((state_all, state), dim=0)

            state_last = state if b == 0 else torch.cat((state_last, state), dim=0)
            state_all = torch.unsqueeze(state_all, dim=0)
            state_batch = state_all if b == 0 else torch.cat((state_batch, state_all), dim=0)

    if direction == 'reverse':
        state_batch = torch.flip(state_batch, [1])

    if list_equal(out_sequence, ['H', 'Hn']):
        self.outputs[0].betensor = state_batch
        self.outputs[1].betensor = state_last
        return (state_batch, state_last)
    elif list_equal(out_sequence, ['Hn']):
        self.outputs[0].betensor = state_last
        return state_last
    else:
        self.outputs[0].betensor = state_batch
        return state_batch


def adaptation_weights_for_unify_shifts_for_aiff(node, weights_name, multiplicator):
    w_ele_t = node.constants[weights_name]
    w_ele = w_ele_t.betensor
    do_scale, do_scale_type, do_shift, do_shift_type = unify_shifts_for_aiff_with_per_n_channel_quant(
        node, w_ele_t, node.attrs["q_mode_weight"], node.attrs["q_bits_weight"], True, lambda xt: multiplicator/xt.scale)
    scale_zp_shape = [w_ele.shape[0]] + [1 for i in range(len(w_ele.shape) - 1)]
    w_scale_expand = w_ele_t.scale.reshape(scale_zp_shape)
    w_zerop_expand = w_ele_t.zerop.reshape(scale_zp_shape)
    quantized_weight = linear_quantize_clip(w_ele, w_scale_expand, w_zerop_expand, w_ele_t.qmin, w_ele_t.qmax)
    return quantized_weight, w_ele_t.scale, w_ele_t.zerop


@quant_register(OpType.GRUv3)
def gruv3_quantize(self, *args):
    q_mode_weight = self.attrs["q_mode_weight"]
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_weight = self.attrs["q_bits_weight"]
    q_bits_bias = self.attrs["q_bits_bias"]
    q_bits_activation = self.attrs["q_bits_activation"]

    dtype = dtype2str(bits2dtype(q_bits_activation, is_signed=True))
    if dtype not in ['int8', 'int16']:
        OPT_FATAL("Currently gruv3/gruv1 only  support quantization bits  of activations is 8 or 16")

    inp = self.inputs[0]
    time_step = self.get_param('time_steps')
    input_size = self.get_param('input_size')
    cell_size = self.get_param('cell_size')
    activations_list = self.get_param('activations', optional=True, default_value=['SIGMOID', 'TANH'])
    activation_alpha = self.get_param('activation_alpha', optional=True, default_value=[1, 1])
    activation_beta = self.get_param('activation_beta', optional=True, default_value=[0, 0])

    h_all = self.placeholders[0]
    h_all.scale, h_all.zerop, h_all.qmin, h_all.qmax, h_all.dtype = \
        get_linear_quant_params_from_tensor(h_all, QuantMode.to_symmetric(
            q_mode_activation), q_bits_activation, is_signed=True)
    h_all.qbits = q_bits_activation
    h_all.qinvariant = False
    h_scale = h_all.scale

    bak_inp_tensor_property = get_bak_tensor_property(self.inputs[0])
    bak_outp_tensor_property = get_bak_tensor_property(self.outputs[0])

    activations_in_scale = []
    activations_out_scale = []
    activation_idx_lut_name = {0: ('lut_rt', 'lut_zt'), 1: ('lut_ht',)}
    for activation_idx, activation in enumerate(activations_list):
        if activation in ["SIGMOID", "TANH"]:  # "SOFTPLUS", "SOFTSIGN", "ELU", "HARDSIGMOID", "SCALEDTANH", "AFFINE", "THRESHOLDEDRELU"
            activation_in_scale, activation_out_scale = \
                generate_lut_with_placeholders(self, activation_idx, activation, q_mode_activation,
                                               q_bits_activation, activation_idx_lut_name, with_lut=True, with_clamp=True, *args)
            if dtype == 'int8':
                activation_in_scale = 256 * activation_in_scale
            activations_in_scale.append(activation_in_scale)
            activations_out_scale.append(activation_out_scale)
        else:
            OPT_FATAL("Don't support activation[%s]." % (activation))

    _tensor_default_property = get_tensor_default_property()
    for p in _tensor_default_property:
        self.inputs[0].__setattr__(p, bak_inp_tensor_property[self.inputs[0].name][p])
        self.outputs[0].__setattr__(p, bak_outp_tensor_property[self.outputs[0].name][p])

    f_in_scale, g_in_scale = activations_in_scale
    f_out_scale, g_out_scale = activations_out_scale

    if self.attrs['unify_shifts_for_aiff'] and QuantMode.is_per_channel(q_mode_weight):
        # quantize wx_gk_t_scale
        quantized_wx_gk, wx_gk_t_scale, wx_gk_t_zerop = adaptation_weights_for_unify_shifts_for_aiff(
            self, 'wx_gk', f_in_scale / inp.scale)
        quantized_wh_gk, wh_gk_t_scale, wh_gk_t_zerop = adaptation_weights_for_unify_shifts_for_aiff(
            self, 'wh_gk', inp.scale * wx_gk_t_scale / h_scale)
        quantized_wx_ck, wx_ck_t_scale, wx_ck_t_zerop = adaptation_weights_for_unify_shifts_for_aiff(
            self, 'wx_ck', g_in_scale / inp.scale)
        quantized_wh_ck, wh_ck_t_scale, wh_ck_t_zerop = adaptation_weights_for_unify_shifts_for_aiff(
            self, 'wh_ck', inp.scale * wx_ck_t_scale / (h_scale * f_out_scale))
        w_scale_expand = [wx_gk_t_scale, wh_gk_t_scale, wx_ck_t_scale, wh_ck_t_scale]
        w_zerop_expand = [wx_gk_t_zerop, wh_gk_t_zerop, wx_ck_t_zerop, wh_ck_t_zerop]
        [self.constants.pop(weights_name) for weights_name in split_weights_name]
    else:
        w_scale_expand = []
        w_zerop_expand = []
        quantized_w = []
        for idx, weights_name in enumerate(split_weights_name):
            w_ele_t = self.constants[weights_name]
            w_ele = w_ele_t.betensor
            w_ele_t.scale, w_ele_t.zerop, w_ele_t.qmin, w_ele_t.qmax, w_ele_t.dtype = \
                get_linear_quant_params_from_tensor(w_ele_t, q_mode_weight, q_bits_weight, is_signed=True)
            if QuantMode.is_per_channel(q_mode_weight):
                w_scale_expand.append(w_ele_t.scale)
                w_zerop_expand.append(w_ele_t.zerop)
                scale_zp_shape = [w_ele.shape[0]] + [1 for i in range(len(w_ele.shape) - 1)]
                quantized_weight = linear_quantize_clip(w_ele, w_scale_expand[idx].reshape(scale_zp_shape),
                                                        w_zerop_expand[idx].reshape(scale_zp_shape), w_ele_t.qmin,
                                                        w_ele_t.qmax)
            else:
                w_scale_expand.append(w_ele_t.scale)
                w_zerop_expand.append(w_ele_t.zerop)
                quantized_weight = linear_quantize_clip(
                    w_ele, w_scale_expand[idx], w_zerop_expand[idx], w_ele_t.qmin, w_ele_t.qmax)
            quantized_w.append(quantized_weight)
            self.constants.pop(weights_name)
        wx_gk_t_scale, wh_gk_t_scale, wx_ck_t_scale, wh_ck_t_scale = w_scale_expand
        wx_gk_t_zerop, wh_gk_t_zerop, wx_ck_t_zerop, wh_ck_t_zerop = w_zerop_expand
        quantized_wx_gk, quantized_wh_gk, quantized_wx_ck, quantized_wh_ck = quantized_w

    gate_kernel_q = torch.cat((quantized_wx_gk, quantized_wh_gk), dim=1)
    cand_kernel_q = torch.cat((quantized_wx_ck, quantized_wh_ck), dim=1)
    quantized_weight = torch.cat((gate_kernel_q, cand_kernel_q), dim=0).contiguous()
    weights = self.constants["weights"]
    weights.scale = torch.cat(w_scale_expand, dim=0)
    weights.zerop = torch.cat(w_zerop_expand, dim=0)
    weights.betensor = quantized_weight
    weights.qbits = q_bits_weight
    weights.dtype = bits2dtype(weights.qbits, is_signed=True)
    weights.qinvariant = False

    xwc_scale = inp.scale * wx_ck_t_scale
    cb_scale = xwc_scale
    # quantized bias
    b = self.constants["biases"]
    bias = b.betensor
    gates_bias = bias[: 2 * cell_size]
    candidate_bias = bias[2 * cell_size: 3 * cell_size]
    xwg_scale = inp.scale * wx_gk_t_scale
    hwg_scale = h_scale * wh_gk_t_scale
    # torch.min(xwg_scale, hwg_scale) if isinstance(xwg_scale, torch.Tensor) else min(xwg_scale, hwg_scale)
    gb_scale = xwg_scale
    zerop = 0
    qmin = -2**(q_bits_bias-1)
    qmax = 2**(q_bits_bias-1) - 1
    gates_bias_q = linear_quantize_clip(gates_bias, gb_scale, zerop, qmin, qmax)

    if 'version' in self.params and self.params['version'] == "GRUV1":
        factor = 1
        hidden_out_placeholders = self.placeholders[5]
        hidden_out_placeholders.scale, hidden_out_placeholders.zerop, \
            hidden_out_placeholders.qmin, hidden_out_placeholders.qmax, hidden_out_placeholders.dtype = \
            get_linear_quant_params_from_tensor(
                hidden_out_placeholders, "per_tensor_symmetric_restricted_range", q_bits_activation, is_signed=True)
        hidden_out_placeholders.qbits = q_bits_activation
        hidden_out_placeholders.qinvariant = False
        hidden_scale, hidden_scale_type, hidden_shift, hidden_shift_type = \
            get_scale_approximation_params(hidden_out_placeholders.scale / (h_scale * wh_ck_t_scale),
                                           q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)
        if QuantMode.is_per_channel(q_mode_weight):
            self.constants["hidden_scale"] = PyTensor(
                self.name + "/hidden_scale", hidden_scale.cpu().numpy().astype(dtype2nptype(hidden_scale_type)))
            self.constants["hidden_scale"].dtype = hidden_scale_type
            self.constants["hidden_shift"] = PyTensor(
                self.name + "/hidden_shift", hidden_shift.cpu().numpy().astype(dtype2nptype(hidden_shift_type)))
            self.constants["hidden_shift"].dtype = hidden_shift_type
        else:
            self.params["hidden_shift_value"] = int(hidden_shift)
            self.params["hidden_shift_type"] = hidden_shift_type
            self.params["hidden_scale_value"] = int(hidden_scale)
            self.params["hidden_scale_type"] = hidden_scale_type

        hidden_bias = bias[3 * cell_size:]
        hidden_bias_scale = h_scale * wh_ck_t_scale
        hidden_bias_q = linear_quantize_clip(hidden_bias, hidden_bias_scale, zerop, qmin, qmax)
        hwc_scale = hidden_scale * f_out_scale
        quantized_bias = hidden_bias_q
    else:
        factor = 2 ** q_bits_activation
        hwc_scale = h_scale * f_out_scale * wh_ck_t_scale
        quantized_bias = torch.zeros([0], device=inp.betensor.device)

    # get_scale_approximation_params = compute_requantization_param
    hwg_do_scale, hwg_do_scale_type, hwg_do_shift, hwg_do_shift_type = \
        get_scale_approximation_params(gb_scale / hwg_scale,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)
    sum_to_sigm_do_scale, sum_to_sigm_do_scale_type, sum_to_sigm_do_shift, sum_to_sigm_do_shift_type = \
        get_scale_approximation_params(f_in_scale / gb_scale,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)

    hwc_do_scale, hwc_do_scale_type, hwc_do_shift, hwc_do_shift_type = \
        get_scale_approximation_params(factor * cb_scale / hwc_scale,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)
    sum_to_tanh_do_scale, sum_to_tanh_do_scale_type, sum_to_tanh_do_shift, sum_to_tanh_do_shift_type = \
        get_scale_approximation_params(g_in_scale / cb_scale,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)
    if dtype == 'int8':
        sum_to_sigm_do_shift += 8
        sum_to_tanh_do_shift += 8

    c1mu_do_scale, c1mu_do_scale_type, c1mu_do_shift, c1mu_do_shift_type = \
        get_scale_approximation_params(h_scale / g_out_scale,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)

    hm1_do_scale, hm1_do_scale_type, hm1_do_shift, hm1_do_shift_type = \
        get_scale_approximation_params(1.0 / f_out_scale,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)

    if QuantMode.is_per_channel(q_mode_weight):
        do_scale = torch.cat((hwg_do_scale, sum_to_sigm_do_scale, hwc_do_scale, sum_to_tanh_do_scale,
                              torch.tensor([c1mu_do_scale], device=inp.betensor.device),
                              torch.tensor([hm1_do_scale], device=inp.betensor.device)))
        do_shift = torch.cat((hwg_do_shift, sum_to_sigm_do_shift, hwc_do_shift, sum_to_tanh_do_shift,
                              torch.tensor([c1mu_do_shift], device=inp.betensor.device),
                              torch.tensor([hm1_do_shift], device=inp.betensor.device)))

        _, do_scale_type = range2dtype(0, do_scale.max().item())  # bits2dtype(q_bits_activation, is_signed=False)
        _, do_shift_type = range2dtype(do_shift.min().item(), do_shift.max().item(), force_int=True)
        self.constants["scale"] = PyTensor(
            self.name+"/scale", do_scale.cpu().numpy().astype(dtype2nptype(do_scale_type)))
        self.constants["scale"].dtype = do_scale_type
        self.constants["shift"] = PyTensor(
            self.name+"/shift", do_shift.cpu().numpy().astype(dtype2nptype(do_shift_type)))
        self.constants["shift"].dtype = do_shift_type
    else:
        do_scale = torch.tensor([hwg_do_scale, sum_to_sigm_do_scale, hwc_do_scale, sum_to_tanh_do_scale,
                                 c1mu_do_scale, hm1_do_scale], device=inp.betensor.device)
        do_shift = torch.tensor([hwg_do_shift, sum_to_sigm_do_shift, hwc_do_shift, sum_to_tanh_do_shift,
                                 c1mu_do_shift, hm1_do_shift], device=inp.betensor.device)
        _, do_scale_type = range2dtype(0, do_scale.max().item())  # bits2dtype(q_bits_activation, is_signed=False)
        _, do_shift_type = range2dtype(do_shift.min().item(), do_shift.max().item(), force_int=True)
        scale_value = do_scale.cpu().numpy().astype(dtype2nptype(do_scale_type)).tolist()
        shift_value = do_shift.cpu().numpy().astype(dtype2nptype(do_shift_type)).tolist()
        self.params["shift_value"] = shift_value
        self.params["shift_type"] = [do_shift_type] * len(shift_value)
        self.params["scale_value"] = scale_value
        self.params["scale_type"] = [do_scale_type] * len(scale_value)

    # torch.min(xwc_scale, hwc_scale) if isinstance(xwc_scale, torch.Tensor) else min(xwc_scale, hwc_scale)  # xwc_scale if xwc_scale <= hwc_scale else hwc_scale
    candidate_bias_q = linear_quantize_clip(candidate_bias, cb_scale, zerop, qmin, qmax)
    quantized_bias = torch.cat((gates_bias_q, candidate_bias_q, quantized_bias), dim=0)
    b.scale = torch.cat([gb_scale, cb_scale], dim=0)
    b.zerop = torch.zeros_like(b.scale)
    b.qmin = qmin
    b.qmax = qmax
    b.qbits = q_bits_bias
    b.betensor = quantized_bias
    b.dtype = bits2dtype(b.qbits, is_signed=True)
    b.qinvariant = False

    if 'remain_shift' in self.attrs:
        self.params['remain_shift'] = self.attrs['remain_shift']

    absorb_input_h_zp_to_bias(self, *args)
    gru_clear_lower_bits_for_bias(self, *args)

    for out in self.outputs:
        out.qbits = q_bits_activation
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, QuantMode.to_symmetric(q_mode_activation), out.qbits, is_signed=True)
        out.qinvariant = False


def gru_clear_lower_bits_for_bias(self, *args, dim=-1):
    if QuantMode.is_per_channel(self.attrs["q_mode_weight"]):
        # AIFF doesn't support per channel weight is not currently supported
        return
    bias = self.constants['biases']
    lmin, lmax = bits2range(self.attrs['bias_effective_bits'], is_signed=True)
    cell_size = self.params['cell_size']
    shift_value = self.params['shift_value']
    aasrb = self.get_param('remain_shift', optional=True, default_value=15)

    gate_bias = bias.betensor[:2 * cell_size]
    candidate_bias = bias.betensor[2 * cell_size: 3 * cell_size]
    bn = None
    if bias.betensor.shape[0] == 4 * cell_size:
        bn = bias.betensor[3 * cell_size:]

    gate_preshift = 0
    candidate_preshift = 0
    bn_preshift = 0
    if dtype2bits(self.inputs[0].dtype) > 8:
        gate_preshift = max(shift_value[1] - aasrb, 0)
        candidate_preshift = max(shift_value[3] - aasrb, 0)
        bn_preshift = max(shift_value[2] - aasrb, 0)

    gate_bias = compress_int32_to_int16(gate_bias, lmin, lmax, gate_preshift)
    candidate_bias = compress_int32_to_int16(candidate_bias, lmin, lmax, candidate_preshift)
    if bn is not None:
        bn = compress_int32_to_int16(bn, lmin, lmax, bn_preshift)

    new_bias = torch.cat([gate_bias, candidate_bias], dim=dim)
    if bn is not None:
        new_bias = torch.cat([new_bias, bn], dim=dim)
    bias.betensor = new_bias
