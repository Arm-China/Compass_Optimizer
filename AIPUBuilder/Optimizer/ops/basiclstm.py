# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.rnn import *
from AIPUBuilder.Optimizer.logger import *
import torch.nn as nn


def qat_basiclstm_forward(self):
    inp0 = self.inputs[0]
    dev = inp0.device

    lut_in_bits = self.inputs[0].qbits
    lut_out_bits = self.outputs[0].qbits

    input_seq = inp0.betensor.float()
    hm1_q_s = self.inputs[1].betensor.float()
    c_prev_s = self.inputs[2].betensor.float()

    batch_size = input_seq.shape[0]

    time_step = self.get_param('time_steps')
    input_size = self.get_param('input_size')
    cell_size = self.get_param('cell_size')
    direction = self.get_param('direction').lower()

    if direction.lower() == 'reverse':
        input_seq = torch.flip(input_seq, [1])

    w = self.constants["weights"].betensor.clone()
    bias = self.constants['biases'].betensor.clone().float()

    weights = w.permute(1, 0).float()
    wx_q = weights[0:input_size, :]
    wh_q = weights[input_size:, :]

    scale_start = 3
    shift_start = 3
    scales = self.constants['scale'].betensor.reshape(time_step, -1)
    shifts = self.constants['shift'].betensor.reshape(time_step, -1)
    zerops = self.constants['zerop'].betensor.reshape(time_step, -1)

    ft_table = self.constants['lut_ft'].betensor.reshape(time_step, -1)
    it_table = self.constants['lut_it'].betensor.reshape(time_step, -1)
    gt_table = self.constants['lut_ct'].betensor.reshape(time_step, -1)
    ot_table = self.constants['lut_ot'].betensor.reshape(time_step, -1)
    h_table = self.constants['lut_h'].betensor.reshape(time_step, -1)

    output = torch.zeros([batch_size, time_step, cell_size]).to(dev)
    output2 = torch.zeros_like(output)

    for batch, data_batch in enumerate(input_seq):
        c_prev = c_prev_s[batch, :]
        hm1_q = hm1_q_s[batch, :]
        for ts in range(time_step):
            scale = scales[ts]
            shift = shifts[ts]
            zerop = zerops[ts]

            x_q = data_batch[ts:ts+1].reshape(1, input_size)
            c_prev_zerop = self.inputs[2].zerop if ts == 0 else zerops[ts-1][6]
            c_prev = c_prev + c_prev_zerop

            hw_do_scale, hw_do_shift = scale[0], shift[0]
            fc_do_scale, fc_do_shift = scale[1], shift[1]
            hout_do_scale, hout_do_shift = scale[2], shift[2]

            x_by_w = torch.matmul(x_q, wx_q)
            h_by_w = torch.matmul(hm1_q, wh_q)

            MIN_INT32, MAX_INT32 = -2**31, 2**31-1
            MIN_INT8, MAX_INT8 = -128, 127
            req_h_by_w = linear_requantize(h_by_w, hw_do_scale, hw_do_shift, 0, MIN_INT32, MAX_INT32)
            mat_sum = linear_requantize(x_by_w + bias + req_h_by_w, 1.0, 0, 0, MIN_INT32, MAX_INT32)
            rescaled_mat_sum = linear_requantize(mat_sum, fc_do_scale, fc_do_shift, zerop[0], MIN_INT8, MAX_INT8)
            i_in, g_in, f_in, o_in = torch.chunk(rescaled_mat_sum, 4, dim=1)

            f = lookup_lut_powerof2(f_in.reshape(-1), ft_table[ts], lut_in_bits, True, lut_out_bits, True)
            i = lookup_lut_powerof2(i_in.reshape(-1), it_table[ts], lut_in_bits, True, lut_out_bits, True)
            g = lookup_lut_powerof2(g_in.reshape(-1), gt_table[ts], lut_in_bits, True, lut_out_bits, True)
            o = lookup_lut_powerof2(o_in.reshape(-1), ot_table[ts], lut_in_bits, True, lut_out_bits, True)

            f_times_c_prev = (f + zerop[2]) * c_prev
            i_times_g = (i + zerop[3]) * (g + zerop[5])
            ig_b_do_scale, fcprev_b_do_scale, scale0, scale1, ts_b_do_scale = scale[scale_start: scale_start + 5]
            ig_b_do_shift, fcprev_b_do_shift, ts_b_do_shift = shift[shift_start: shift_start + 3]

            res_i_times_g = linear_requantize(i_times_g, ig_b_do_scale, ig_b_do_shift, 0, MIN_INT32, MAX_INT32)
            res_i_times_g_times_scale0 = linear_requantize(res_i_times_g, scale0, 0, 0, MIN_INT32, MAX_INT32)
            res_f_times_c_prev = linear_requantize(f_times_c_prev, fcprev_b_do_scale, fcprev_b_do_shift, zerop[10],
                                                   MIN_INT8, MAX_INT8)
            c_tmp = linear_requantize((res_i_times_g_times_scale0 + (res_f_times_c_prev + zerop[10]) * scale1),
                                      ts_b_do_scale, ts_b_do_shift, 0, MIN_INT32, MAX_INT32)

            re_scaled_c_tmp = linear_requantize(c_tmp, 1., 0, zerop[6], MIN_INT8, MAX_INT8)  # c_lut no need consider zp
            c_prev = re_scaled_c_tmp

            c_lut_out = lookup_lut_powerof2(re_scaled_c_tmp.reshape(-1),
                                            h_table[ts], lut_in_bits, True, lut_out_bits, True)

            hm1_q_tmp = (o + zerop[4]) * (c_lut_out + zerop[7])
            h_prev_new = linear_requantize(hm1_q_tmp, hout_do_scale, hout_do_shift, zerop[8], MIN_INT8, MAX_INT8)
            hm1_q = h_prev_new

            output[batch][ts] = h_prev_new
            output2[batch][ts] = re_scaled_c_tmp

    if direction.lower() == 'reverse':
        output = torch.flip(output, [1])

    self.outputs[0].betensor = output
    if len(self.outputs) == 2:
        self.outputs[1].betensor = output2


@op_register(OpType.BasicLSTM)
def lstm(self, *args):
    is_for_qat = self.get_param('basiclstm_for_qat', optional=True, default_value=False)
    if is_for_qat:
        qat_basiclstm_forward(self)
        return [o.betensor for o in self.outputs]

    inp0 = self.inputs[0]
    input_seq = inp0.betensor.double()
    h_cell = self.inputs[1].betensor.double()
    c_cell = self.inputs[2].betensor.double()
    batch_size = input_seq.shape[0]
    time_step = self.get_param('time_steps')
    input_size = self.get_param('input_size')
    h_initial_batch = h_cell.shape[0]
    c_initial_batch = c_cell.shape[0]
    cell_size = self.get_param('cell_size')
    direction = self.get_param('direction')

    if direction == 'reverse':
        input_seq = torch.flip(input_seq, [1])

    start_data_idx = self.current_batch_idx * batch_size
    start_h_initial_idx = start_data_idx % h_initial_batch
    start_c_initial_idx = start_data_idx % c_initial_batch

    h_initial = h_cell[start_h_initial_idx: start_h_initial_idx + 1]
    c_initial = c_cell[start_c_initial_idx: start_c_initial_idx + 1]
    for initial_idx in range(1, batch_size):
        current_h_initial_idx = (start_data_idx + initial_idx) % h_initial_batch
        current_c_initial_idx = (start_data_idx + initial_idx) % c_initial_batch
        h_initial = torch.cat((h_initial, h_cell[current_h_initial_idx: current_h_initial_idx + 1]), dim=0)
        c_initial = torch.cat((c_initial, c_cell[current_c_initial_idx: current_c_initial_idx + 1]), dim=0)

    w = self.constants["weights"].betensor.clone()
    bias = self.constants['biases'].betensor.clone().double()
    dev = inp0.device
    if self.quantized:
        w += self.constants["weights"].broadcast_zerop
        bias += self.constants['biases'].broadcast_zerop
    weights = w.permute(1, 0).double()
    out_sequence = self.get_param('out_sequence')
    activations_list = self.get_param('activations') \
        if 'activations' in self.params else ['SIGMOID', 'TANH', 'TANH']

    h_batch = torch.zeros([batch_size, time_step, cell_size], device=dev)
    h_last = torch.zeros([batch_size, cell_size], device=dev)
    c_batch = torch.zeros([batch_size, time_step, cell_size], device=dev)
    c_last = torch.zeros([batch_size, cell_size], device=dev)

    if not self.quantized:
        threshold = self.get_param('threshold', optional=True, default_value=float('inf'))
        cell_clip = self.get_param('cell_clip', optional=True, default_value=float('inf'))
        forget_bias = self.get_param('forget_bias', optional=True, default_value=0.0)

        f_lut_in = torch.zeros([batch_size, time_step, 3*cell_size], device=dev)
        f_lut_out = torch.zeros([batch_size, time_step, 3*cell_size], device=dev)
        g_lut_in = torch.zeros([batch_size, time_step, cell_size], device=dev)
        g_lut_out = torch.zeros([batch_size, time_step, cell_size], device=dev)
        h_lut_in = torch.zeros([batch_size, time_step, cell_size], device=dev)
        h_lut_out = torch.zeros([batch_size, time_step, cell_size], device=dev)

        for b in range(batch_size):
            h_all = torch.zeros([1, cell_size], device=dev, dtype=torch.float)
            c_all = torch.zeros([1, cell_size], device=dev, dtype=torch.float)
            h_prev = torch.unsqueeze(h_initial[b], dim=0)
            c_prev = torch.unsqueeze(c_initial[b], dim=0)
            for ts in range(time_step):
                in_ts = torch.reshape(input_seq[b, ts, :], (-1, input_size))
                inputs = torch.cat((in_ts, h_prev), dim=1)
                sum0 = torch.add(torch.matmul(inputs, weights), torch.unsqueeze(bias, 0))
                i_tmp, g_tmp, f_tmp, o_tmp = torch.chunk(sum0, 4, dim=1)
                i_tmp = torch.clamp(i_tmp, -threshold, threshold)
                g_tmp = torch.clamp(g_tmp, -threshold, threshold)
                f_tmp = torch.clamp(f_tmp + forget_bias, -threshold, threshold)
                o_tmp = torch.clamp(o_tmp, -threshold, threshold)

                f_lut_in[b, ts, :] = torch.squeeze(torch.cat((i_tmp, f_tmp, o_tmp), dim=1), 0)
                g_lut_in[b, ts, :] = torch.squeeze(g_tmp, 0)

                i = g_rnn_activation_func[activations_list[0]][1](i_tmp)
                g = g_rnn_activation_func[activations_list[1]][1](g_tmp)
                f = g_rnn_activation_func[activations_list[0]][1](f_tmp)
                o = g_rnn_activation_func[activations_list[0]][1](o_tmp)

                f_lut_out[b, ts, :] = torch.squeeze(torch.cat((i, f, o), dim=1), 0)
                g_lut_out[b, ts, :] = torch.squeeze(g, 0)

                c_prev = torch.multiply(f, c_prev) + torch.multiply(i, g)
                c_prev = torch.clamp(c_prev, -cell_clip, cell_clip)
                h_lut_in[b, ts, :] = torch.squeeze(c_prev, 0)
                c_lut = g_rnn_activation_func[activations_list[2]][1](c_prev)
                h_lut_out[b, ts, :] = torch.squeeze(c_lut, 0)
                h_prev = torch.multiply(o, c_lut)

                h_all = h_prev if ts == 0 else torch.cat((h_all, h_prev), dim=0)
                c_all = c_prev if ts == 0 else torch.cat((c_all, c_prev), dim=0)

            # h_expd = torch.unsqueeze(h_prev, dim = 0)
            h_last = h_prev if b == 0 else torch.cat((h_last, h_prev), dim=0)

            h_all = torch.unsqueeze(h_all, dim=0)
            h_batch = h_all if b == 0 else torch.cat((h_batch, h_all), dim=0)

            # c_expd = torch.unsqueeze(c_prev, dim = 0)
            c_last = c_prev if b == 0 else torch.cat((c_last, c_prev), dim=0)

            c_all = torch.unsqueeze(c_all, dim=0)
            c_batch = c_all if b == 0 else torch.cat((c_batch, c_all), dim=0)

        # currently AIFF only use two activation lut
        if activations_list[0] == activations_list[1]:
            equal_activation_lut_in = torch.cat((f_lut_in, g_lut_in), dim=2)
            equal_activation_lut_out = torch.cat((f_lut_out, g_lut_out), dim=2)
            f_lut_in = g_lut_in = equal_activation_lut_in
            f_lut_out = g_lut_out = equal_activation_lut_out
        if activations_list[0] == activations_list[2]:
            equal_activation_lut_in = torch.cat((f_lut_in, h_lut_in), dim=2)
            equal_activation_lut_out = torch.cat((f_lut_out, h_lut_out), dim=2)
            f_lut_in = h_lut_in = equal_activation_lut_in
            f_lut_out = h_lut_out = equal_activation_lut_out
        if activations_list[1] == activations_list[2]:
            equal_activation_lut_in = torch.cat((g_lut_in, h_lut_in), dim=2)
            equal_activation_lut_out = torch.cat((g_lut_out, h_lut_out), dim=2)
            g_lut_in = h_lut_in = equal_activation_lut_in
            g_lut_out = h_lut_out = equal_activation_lut_out

        lut_placeholder_list = ['h_batch', 'f_lut_in', 'f_lut_out', 'g_lut_in', 'g_lut_out', 'h_lut_in', 'h_lut_out']
        if len(self.placeholders) < 1:
            for placeholder_name in lut_placeholder_list:
                ph = PyTensor(self.name + '/' + placeholder_name,
                              eval(placeholder_name).cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
                self.placeholders.append(ph)
            for c_ts in range(c_batch.shape[1]):
                ph = PyTensor(self.name + "/c_state_" + str(c_ts),
                              c_batch[:, c_ts, :].cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
                self.placeholders.append(ph)
        for idx, placeholder_name in enumerate(lut_placeholder_list):
            self.placeholders[idx].betensor = eval(placeholder_name)
        for c_ts in range(c_batch.shape[1]):
            self.placeholders[7 + c_ts].betensor = c_batch[:, c_ts, :]

    else:
        dtype = dtype2str(self.outputs[0].dtype)
        q_bits_activation = dtype2bits(self.outputs[0].dtype)
        if dtype not in ['int8', 'int16']:
            OPT_FATAL("Currently gruv3/gruv1 only  support quantization bits  of activations is 8 or 16")
        qmin, qmax = bits2range(q_bits_activation, True)

        wx_q = weights[0:input_size, :]
        wh_q = weights[input_size:, :]

        it_table = self.constants['lut_it'].betensor
        ft_table = self.constants['lut_ft'].betensor
        ct_table = self.constants['lut_ct'].betensor
        ot_table = self.constants['lut_ot'].betensor
        h_table = self.constants['lut_h'].betensor
        lut_in_bits = self.inputs[0].qbits
        lut_out_bits = self.outputs[0].qbits

        # Temporarily set it to zero because C quantization method is symmetric
        c_zerop = torch.zeros(2 * time_step, device=dev)

        scale_ = self.constants["scale"].betensor
        shift_ = self.constants["shift"].betensor
        diff_shifts_ = self.constants["diff_shifts"].betensor
        if is_torch_tensor_with_multi_data(self.constants["weights"].scale):
            scale = scale_.to(dev)
            shift = shift_.to(dev)
            scale_ = []
            shift_ = []
            step_length = [1, cell_size * 4, cell_size, 1, 1]
            start_idx = 0
            for idx, step in enumerate(step_length):
                scale_.append(scale[start_idx: start_idx + step])
                shift_.append(shift[start_idx: start_idx + step])
                start_idx += step

            it_scale, it_shift = scale_[1][:cell_size], shift_[1][:cell_size]
            ft_scale, ft_shift = scale_[1][2 * cell_size:3 * cell_size], shift_[1][2 * cell_size:3 * cell_size]
            ct_scale, ct_shift = scale_[2][:cell_size], shift_[2][:cell_size]
            ot_scale, ot_shift = scale_[1][3 * cell_size:], shift_[1][3 * cell_size:]

            for ts in range(time_step):
                scale_.extend([scale[start_idx + 2 * ts], scale[start_idx + 2 * ts + 1]])
                shift_.extend([shift[start_idx + 2 * ts], shift[start_idx + 2 * ts + 1]])

        else:
            it_scale, it_shift = scale_[1], shift_[1]
            ft_scale, ft_shift = scale_[1], shift_[1]
            ct_scale, ct_shift = scale_[2], shift_[2]
            ot_scale, ot_shift = scale_[1], shift_[1]

        act_qmax = 2 ** 31 - 1
        act_qmin = -2 ** 31
        for b in range(batch_size):
            h_all = torch.zeros([1, cell_size], device=dev, dtype=torch.double)
            c_all = torch.zeros([1, cell_size], device=dev, dtype=torch.double)
            h_prev = torch.unsqueeze(h_initial[b], dim=0).double()
            c_prev = torch.unsqueeze(c_initial[b], dim=0).double()
            for ts in range(time_step):
                in_ts = torch.reshape(input_seq[b, ts, :], (-1, input_size)).double()
                x_by_wx = torch.matmul(in_ts, wx_q)
                h_by_wh = torch.matmul(h_prev.to(wh_q.dtype), wh_q)
                if dtype == 'int8':
                    re_scaled_h_by_wh = linear_requantize(h_by_wh, scale_[0], shift_[0], 0, act_qmin, act_qmax)
                    mat_sum = x_by_wx + re_scaled_h_by_wh + bias
                    i_tmp, g_tmp, f_tmp, o_tmp = torch.chunk(mat_sum, 4, dim=1)

                    f_in = linear_requantize(f_tmp, ft_scale, ft_shift, 0, qmin, qmax)
                    i_in = linear_requantize(i_tmp, it_scale, it_shift, 0, qmin, qmax)
                    g_in = linear_requantize(g_tmp, ct_scale, ct_shift, 0, qmin, qmax)
                    o_in = linear_requantize(o_tmp, ot_scale, ot_shift, 0, qmin, qmax)

                    # g_rnn_activation_func[activations_list[0]][2](f_in, ft_table).float()
                    f = lookup_lut_powerof2(f_in, ft_table, lut_in_bits, True, lut_out_bits, True)
                    # g_rnn_activation_func[activations_list[0]][2](i_in, it_table).float()
                    i = lookup_lut_powerof2(i_in, it_table, lut_in_bits, True, lut_out_bits, True)
                    # g_rnn_activation_func[activations_list[1]][2](g_in, ct_table).float()
                    g = lookup_lut_powerof2(g_in, ct_table, lut_in_bits, True, lut_out_bits, True)
                    # g_rnn_activation_func[activations_list[0]][2](o_in, ot_table).float()
                    o = lookup_lut_powerof2(o_in, ot_table, lut_in_bits, True, lut_out_bits, True)

                elif dtype == 'int16':
                    m_shift = self.get_param('lut_shift_value')
                    h_by_wh_s = linear_requantize(h_by_wh, 1, shift_[0] - m_shift, 0, act_qmin, act_qmax)
                    re_scaled_h_by_wh_s = linear_requantize(h_by_wh_s, scale_[0], shift_[1], 0, act_qmin, act_qmax)
                    x_by_wx_s = linear_requantize(x_by_wx, 1, shift_[1] - m_shift, 0, act_qmin, act_qmax)
                    bias_s = linear_requantize(bias, 1, shift_[1] - m_shift, 0, act_qmin, act_qmax)
                    mat_sum = x_by_wx_s + re_scaled_h_by_wh_s + bias_s
                    mat_sum = torch.clamp(torch.round(mat_sum), act_qmin, act_qmax)
                    i_tmp, g_tmp, f_tmp, o_tmp = torch.chunk(mat_sum, 4, dim=1)

                    f_in = linear_requantize(f_tmp, ft_scale, m_shift, 0, qmin, qmax)
                    i_in = linear_requantize(i_tmp, it_scale, m_shift, 0, qmin, qmax)
                    g_in = linear_requantize(g_tmp, ct_scale, ct_shift - it_shift + m_shift, 0, qmin, qmax)
                    o_in = linear_requantize(o_tmp, ot_scale, m_shift, 0, qmin, qmax)

                    # g_rnn_activation_func[activations_list[0]][3](f_in, ft_table).float()
                    f = lookup_lut_powerof2(f_in, ft_table, lut_in_bits, True, lut_out_bits, True).double()
                    # g_rnn_activation_func[activations_list[0]][3](i_in, it_table).float()
                    i = lookup_lut_powerof2(i_in, it_table, lut_in_bits, True, lut_out_bits, True).double()
                    # g_rnn_activation_func[activations_list[1]][3](g_in, ct_table).float()
                    g = lookup_lut_powerof2(g_in, ct_table, lut_in_bits, True, lut_out_bits, True).double()
                    # g_rnn_activation_func[activations_list[0]][3](o_in, ot_table).float()
                    o = lookup_lut_powerof2(o_in, ot_table, lut_in_bits, True, lut_out_bits, True).double()

                diff_shift = diff_shifts_[ts]
                i_times_g = torch.multiply(i, g)
                i_times_g = linear_requantize(i_times_g, 1, diff_shift, 0, act_qmin, act_qmax)
                f_times_c_prev = torch.multiply(f, c_prev + c_zerop[2 * ts])
                rescaled_f_times_c_prev = linear_requantize(f_times_c_prev, scale_[5 + 2 * ts], shift_[5 + 2 * ts]+diff_shift, 0,
                                                            act_qmin, act_qmax)
                c_tmp = i_times_g + rescaled_f_times_c_prev

                c_tmp = torch.clamp(c_tmp, act_qmin, act_qmax)

                re_scaled_c_tmp = linear_requantize(c_tmp, scale_[3], shift_[3]-diff_shift, 0, qmin+1, qmax)
                ctmp_lut_out = lookup_lut_powerof2(re_scaled_c_tmp, h_table, lut_in_bits, True, lut_out_bits, True)
                h_prev = torch.multiply(o, ctmp_lut_out).double()
                h_prev = linear_requantize(h_prev, scale_[4], shift_[4], 0, qmin+1, qmax)
                c_prev = linear_requantize(c_tmp, scale_[5 + 2 * ts + 1], shift_[5 + 2 * ts + 1]-diff_shift,
                                           c_zerop[2 * ts + 1], qmin+1, qmax)

                h_all = h_prev if ts == 0 else torch.cat((h_all, h_prev), dim=0)
                c_all = c_prev if ts == 0 else torch.cat((c_all, c_prev), dim=0)

            # h_expd = torch.unsqueeze(h_prev, dim = 0) #[1,1,512]
            h_last = h_prev if b == 0 else torch.cat((h_last, h_prev), dim=0)

            h_all = torch.unsqueeze(h_all, dim=0)  # [1,40,512]
            h_batch = h_all if b == 0 else torch.cat((h_batch, h_all), dim=0)

            # c_expd = torch.unsqueeze(c_prev, dim = 0) #[1,1,512]
            c_last = c_prev if b == 0 else torch.cat((c_last, c_prev), dim=0)

            c_all = torch.unsqueeze(c_all, dim=0)  # [1,40,512]
            c_batch = c_all if b == 0 else torch.cat((c_batch, c_all), dim=0)

    results = []
    for idx, sequence in enumerate(out_sequence):
        if 'Y' == sequence:
            if direction == 'reverse':
                h_batch = torch.flip(h_batch, [1])
            self.outputs[idx].betensor = h_batch
            results.append(h_batch)
        elif 'H' == sequence:
            self.outputs[idx].betensor = h_last
            results.append(h_last)
        elif 'C' == sequence:
            self.outputs[idx].betensor = c_last
            results.append(c_last)
    return results


def lstm_clear_lower_bits_for_bias(self, *args, dim=-1):
    if QuantMode.is_per_channel(self.attrs["q_mode_weight"]):
        # AIFF doesn't support per channel weight is not currently
        return
    bias = self.constants['biases']
    shift = self.constants["shift"].betensor
    lmin, lmax = bits2range(self.attrs['bias_effective_bits'], is_signed=True)
    cell_size = self.params['cell_size']

    bias_i, bias_g, bias_f, bias_o = torch.split(bias.betensor, cell_size, dim=dim)
    bias_ifo = torch.cat([bias_i, bias_f, bias_o], dim=dim)
    if dtype2bits(self.inputs[0].dtype) <= 8:
        bias_ifo = compress_int32_to_int16(bias_ifo, lmin, lmax, 0)
        bias_g = compress_int32_to_int16(bias_g, lmin, lmax, 0)
    else:
        bias_ifo = compress_int32_to_int16(bias_ifo, lmin, lmax, shift[1] - self.params['lut_shift_value'])
        bias_g = compress_int32_to_int16(bias_g, lmin, lmax, shift[1] - self.params['lut_shift_value'])

    bias_i, bias_f, bias_o = torch.split(bias_ifo, cell_size, dim=dim)
    new_bias = torch.cat([bias_i, bias_g, bias_f, bias_o], dim=dim)
    bias.betensor = new_bias


@quant_register(OpType.BasicLSTM)
def lstm_quantize(self, *args):
    def hwa_zy_clamp_lut(lut):
        lut[0] = lut[1]
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
    forget_bias = self.get_param('forget_bias', optional=True, default_value=0.0)
    activations_list = self.get_param('activations') if 'activations' in self.params else ['SIGMOID', 'TANH', 'TANH']
    cell_size = self.get_param('cell_size')
    dev = inp.device

    w = self.constants["weights"]
    b = self.constants["biases"]
    zerop = []
    bias = b.betensor
    bias[cell_size * 2: cell_size * 3] += forget_bias

    bak_inp_tensor_property = get_bak_tensor_property(self.inputs[0])
    bak_outp_tensor_property = get_bak_tensor_property(self.outputs[0])

    activations_in_scale = []
    activations_out_scale = []
    activation_idx_lut_name = {0: ('lut_it', 'lut_ft', 'lut_ot'), 1: ('lut_ct',), 2: ('lut_h',)}
    for activation_idx, activation in enumerate(activations_list):
        if activation in ["SIGMOID", "TANH"]:  # , "SOFTPLUS", "SOFTSIGN", "ELU", "HARDSIGMOID", "SCALEDTANH", "AFFINE", "THRESHOLDEDRELU"
            activation_in_scale, activation_out_scale = \
                generate_lut_with_placeholders(self, activation_idx, activation, q_mode_activation,
                                               q_bits_activation, activation_idx_lut_name, with_lut=True, with_clamp=True, *args)

            if dtype == 'int8':
                activation_in_scale = 256 * activation_in_scale
                # consistent with the implementation of lib
                for lut_name in activation_idx_lut_name[activation_idx]:
                    lut = self.constants[lut_name]
                    hwa_zy_clamp_lut(lut.betensor)
            activations_in_scale.append(activation_in_scale)
            activations_out_scale.append(activation_out_scale)
        else:
            OPT_FATAL("Don't support activation[%s]." % (activation))

    _tensor_default_property = get_tensor_default_property()
    for p in _tensor_default_property:
        self.inputs[0].__setattr__(p, bak_inp_tensor_property[self.inputs[0].name][p])
        self.outputs[0].__setattr__(p, bak_outp_tensor_property[self.outputs[0].name][p])

    f_in_scale, g_in_scale, h_in_scale = activations_in_scale
    f_out_scale, g_out_scale, h_out_scale = activations_out_scale

    if QuantMode.is_per_channel(q_mode_weight):
        w.scale, w.zerop, w.qmin, w.qmax, w.dtype = get_linear_quant_params_from_tensor(w, q_mode_weight, q_bits_weight,
                                                                                        is_signed=True)
        w_scale_expand = list(w.scale.cpu().numpy())
        w_zerop_expand = list(w.zerop.cpu().numpy())
        scale_zp_shape = [4 * cell_size, 1]
    else:
        w_out_cnum = w.shape[0]
        w_scale_expand = []
        w_zerop_expand = []
        w.scale, w.zerop, w.qmin, w.qmax, w.dtype = get_linear_quant_params_from_tensor(w, q_mode_weight, q_bits_weight,
                                                                                        is_signed=True)
        w_scale_expand.extend([w.scale] * w_out_cnum)
        w_zerop_expand.extend([w.zerop] * w_out_cnum)
        scale_zp_shape = [w.betensor.shape[0]] + [1 for i in range(len(w.betensor.shape) - 1)]
    w_scale_expand = torch.tensor(w_scale_expand, device=dev)
    w_zerop_expand = torch.tensor(w_zerop_expand, device=dev)
    w.betensor = linear_quantize_clip(w.betensor, w_scale_expand.reshape(scale_zp_shape),
                                      w_zerop_expand.reshape(scale_zp_shape), w.qmin, w.qmax)
    w.qbits = q_bits_weight
    w.qinvariant = False

    xw_scale = inp.scale * w.scale

    # quantize placeholders
    h_all = self.placeholders[0]
    h_all.scale, h_all.zerop, h_all.qmin, h_all.qmax, h_all.dtype = \
        get_linear_quant_params_from_tensor(h_all, QuantMode.to_symmetric(
            q_mode_activation), q_bits_activation, is_signed=True)
    h_all.qbits = q_bits_activation
    h_all.qinvariant = False
    h_scale = h_all.scale

    hw_do_scale, hw_do_scale_type, hw_do_shift, hw_do_shift_type = \
        get_scale_approximation_params(inp.scale / h_scale,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)
    sum_to_sigm_do_scale, sum_to_sigm_do_scale_type, sum_to_sigm_do_shift, sum_to_sigm_do_shift_type = \
        get_scale_approximation_params(f_in_scale / xw_scale,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)

    sum_to_tanh_do_scale, sum_to_tanh_do_scale_type, sum_to_tanh_do_shift, sum_to_tanh_do_shift_type = \
        get_scale_approximation_params(g_in_scale / xw_scale,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)

    c_tanh_do_scale, c_tanh_do_scale_type, c_tanh_do_shift, c_tanh_do_shift_type = \
        get_scale_approximation_params(h_in_scale / f_out_scale / g_out_scale,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)
    lut_shift = 0
    if dtype == 'int8':
        sum_to_sigm_do_shift += 8
        sum_to_tanh_do_shift += 8
        c_tanh_do_shift += 8
    elif dtype == 'int16':
        shift0 = torch.tensor([hw_do_shift], device=dev)
        shift1 = sum_to_sigm_do_shift if isinstance(sum_to_sigm_do_shift, torch.Tensor) else \
            torch.tensor([sum_to_sigm_do_shift], device=dev)
        shift2 = torch.tensor([15], device=dev)
        lut_shift = torch.cat((shift0, shift1, shift2)).min().item()

    hout_do_scale, hout_do_scale_type, hout_do_shift, hout_do_shift_type = \
        get_scale_approximation_params(h_scale / f_out_scale / h_out_scale,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)

    if QuantMode.is_per_channel(q_mode_weight):
        scales = torch.cat((torch.tensor([hw_do_scale], device=dev),
                            sum_to_sigm_do_scale,
                            sum_to_tanh_do_scale[cell_size: 2 * cell_size],
                            torch.tensor([c_tanh_do_scale], device=dev),
                            torch.tensor([hout_do_scale], device=dev)))
        shifts = torch.cat((torch.tensor([hw_do_shift], device=dev),
                            sum_to_sigm_do_shift,
                            sum_to_tanh_do_shift[cell_size: 2 * cell_size],
                            torch.tensor([c_tanh_do_shift], device=dev),
                            torch.tensor([hout_do_shift], device=dev)))
    else:
        scales = torch.tensor([hw_do_scale, sum_to_sigm_do_scale, sum_to_tanh_do_scale, c_tanh_do_scale, hout_do_scale],
                              device=dev)
        shifts = torch.tensor([hw_do_shift, sum_to_sigm_do_shift, sum_to_tanh_do_shift, c_tanh_do_shift, hout_do_shift],
                              device=dev)

    c_do_scales = torch.zeros(2 * time_step, device=dev)
    c_do_shifts = torch.zeros(2 * time_step, device=dev)
    # c_do_zerop = torch.zeros(2 * time_step, device=dev)

    c_scale = torch.zeros(time_step + 1)
    diff_shifts = torch.zeros(time_step)
    # c_zerop = torch.zeros(time_step + 1)
    # c_all = self.placeholders[1].betensor
    c_scale[0] = self.inputs[2].scale

    # c_zerop = self.inputs[2].zerop
    for t in range(time_step):
        c_timestep_t = self.placeholders[7+t]  # c_all[:, t, :]
        diff_shift = 0
        c_timestep_t.scale, c_timestep_t.zerop, c_timestep_t.qmin, c_timestep_t.qmax, c_timestep_t.dtype = \
            get_linear_quant_params_from_tensor(c_timestep_t, QuantMode.to_symmetric(
                q_mode_activation), q_bits_activation, is_signed=True)
        c_scale[t + 1] = c_timestep_t.scale
        c_timestep_t.qbits = q_bits_activation
        c_timestep_t.qinvariant = False
        # c_zerop[t + 1] = c_timestep_t.zerop
        # c_prev_do_zerop = c_zerop
        # c_out_do_zerop = c_timestep_t.zerop
        # c_zerop = c_timestep_t.zerop
        c_prev_do_scale, c_prev_do_scale_type, c_prev_do_shift, c_prev_do_scale_type = \
            get_scale_approximation_params(g_out_scale / c_scale[t],
                                           q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)

        c_out_do_scale, c_out_do_scale_type, c_out_do_shift, c_out_do_shift_type = \
            get_scale_approximation_params(c_timestep_t.scale / f_out_scale / g_out_scale,
                                           q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)

        if dtype == 'int16':
            if c_prev_do_shift < 15:
                diff_shift = 15 - c_prev_do_shift
                diff_shift = min(diff_shift, c_out_do_shift, c_tanh_do_shift)
                diff_shifts[t] = diff_shift

        c_do_scales[2 * t] = c_prev_do_scale
        c_do_shifts[2 * t] = c_prev_do_shift
        # c_do_zerop[2 * t] = c_prev_do_zerop
        c_do_scales[2 * t + 1] = c_out_do_scale
        c_do_shifts[2 * t + 1] = c_out_do_shift
        # c_do_zerop[2 * t + 1] = c_out_do_zerop

    do_scale = torch.cat((scales, c_do_scales), dim=0)
    do_shift = torch.cat((shifts, c_do_shifts), dim=0)

    _, do_scale_type = range2dtype(0, do_scale.max().item())
    _, do_shift_type = range2dtype(do_shift.min(), do_shift.max().item(), force_int=True)
    self.constants["scale"] = PyTensor(self.name + "/scale", do_scale.cpu().numpy().astype(dtype2nptype(do_scale_type)))
    self.constants["scale"].dtype = do_scale_type
    self.constants["shift"] = PyTensor(self.name + "/shift", do_shift.cpu().numpy().astype(dtype2nptype(do_shift_type)))
    self.constants["shift"].dtype = do_shift_type
    self.constants["diff_shifts"] = PyTensor(
        self.name + "/diff_shifts", diff_shifts.cpu().numpy().astype(dtype2nptype(do_shift_type)))
    self.constants["diff_shifts"].dtype = do_shift_type
    self.params["lut_shift_value"] = int(lut_shift)
    self.params["lut_shift_type"] = SHIFT_DTYPE
    # self.constants["zerop"] = PyTensor(self.name + "/zerop", c_do_zerop.cpu().numpy().astype(
    #     dtype2nptype(bits2dtype(q_bits_activation, is_signed=True))))
    # self.constants["zerop"].dtype = bits2dtype(q_bits_activation, is_signed=True)
    # self.params['h_zerop'] = int(h_all.zerop)

    # quantized bias
    b.zerop = 0
    b.scale = xw_scale
    b.qmin = -2 ** (q_bits_bias - 1)
    b.qmax = 2 ** (q_bits_bias - 1) - 1
    b.qbits = q_bits_bias
    b.betensor = linear_quantize_clip(bias, xw_scale, b.zerop, b.qmin, b.qmax)
    b.dtype = bits2dtype(b.qbits, is_signed=True)
    b.qinvariant = False

    absorb_input_h_zp_to_bias(self, *args)
    lstm_clear_lower_bits_for_bias(self, *args)

    for out in self.outputs:
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, QuantMode.to_symmetric(q_mode_activation), q_bits_activation, is_signed=True)
        out.qbits = q_bits_activation
        out.qinvariant = False
