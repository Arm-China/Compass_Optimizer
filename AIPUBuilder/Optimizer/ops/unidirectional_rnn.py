# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.rnn import *
import torch.nn as nn


register_optype('RNN')

split_weights_name = ['wx', 'wh']


@op_register(OpType.RNN)
def rnn(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    outp = self.outputs[0]

    current_batch_idx = self.current_batch_idx
    input_seq = inp0.betensor
    [batch_size, time_step, input_size] = input_seq.shape
    initial_batch = self.inputs[1].ir_shape[0]

    initial_H = inp1.betensor
    start_data_idx = self.current_batch_idx * batch_size

    start_initial_idx = start_data_idx % initial_batch
    in_state = initial_H[start_initial_idx: start_initial_idx + 1]
    for initial_idx in range(1, batch_size):
        current_initial_idx = (start_data_idx + initial_idx) % initial_batch
        in_state = torch.cat((in_state, initial_H[current_initial_idx: current_initial_idx + 1]), dim=0)

    activations = self.get_param('activations')
    cell_size = self.get_param('cell_size')
    direction = self.get_param('direction')
    threshold = self.get_param('threshold', optional=True, default_value=None)

    if direction == "reverse":
        input_seq = torch.flip(input_seq, [1])

    w = self.constants["weights"].betensor
    bias = self.constants['biases'].betensor
    weights = w.permute(1, 0).float()
    wx = weights[:input_size, :]
    wh = weights[input_size:, :]

    # generate output
    state_batch = torch.zeros([batch_size, time_step, cell_size], device=inp0.betensor.device)
    state_last = torch.zeros([batch_size, cell_size], device=inp0.betensor.device)
    xw_hw_sum = torch.zeros([batch_size, time_step, cell_size], device=inp0.betensor.device)
    lut_in_batch = torch.zeros([batch_size, time_step, cell_size], device=inp0.betensor.device)

    if not self.quantized:
        for idx, weights_name in enumerate(split_weights_name):
            if weights_name not in self.constants:
                split_weights = eval(weights_name).permute(1, 0)
                self.constants[weights_name] = PyTensor(
                    self.name+"/constants"+str(idx), split_weights, dtype=Dtype.FP32)
                self.constants[weights_name].betensor = split_weights
                self.constants[weights_name].ir_dtype = self.constants["weights"].dtype
                self.constants[weights_name].dtype = self.constants["weights"].dtype
                self.constants[weights_name].ir_shape = TensorShape(list(split_weights.shape))
        for b in range(batch_size):
            state = torch.unsqueeze(in_state[b], dim=0).float()  # (1,2161)
            state_all = torch.zeros([0, cell_size], device=inp0.betensor.device)
            for ts in range(time_step):
                in_ts = torch.reshape(input_seq[b, ts, :], (-1, input_size)).float()
                #x*wx + h*wh + bias
                x_wx_sum = torch.matmul(in_ts, wx)
                h_wh_sum = torch.matmul(state, wh)
                xw_hw_sum = x_wx_sum + h_wh_sum + torch.unsqueeze(bias, 0)
                if threshold is not None:
                    xw_hw_sum = torch.clamp(xw_hw_sum, -threshold, threshold)
                lut_in_batch[b, ts, :] = torch.squeeze(xw_hw_sum).clone()

                if activations == 'TANH':
                    state = torch.tanh(xw_hw_sum)
                elif activations == 'SIGMOID':
                    state = torch.sigmoid(xw_hw_sum)
                elif activations == 'RELU':
                    state = torch.relu(xw_hw_sum)
                elif activations == 'CLIP':
                    clip_min = self.get_param('clip_min')
                    clip_max = self.get_param('clip_max')
                    state = torch.clamp(xw_hw_sum, clip_min, clip_max)
                elif activations == 'SIGN_BIT':
                    less_zero = xw_hw_sum < 0
                    larger_zero = xw_hw_sum >= 0
                    xw_hw_sum[less_zero] = 1
                    xw_hw_sum[larger_zero] = 0
                    state = xw_hw_sum
                elif activations == 'NONE':
                    state = xw_hw_sum

                state_all = torch.cat((state_all, state), dim=0)

            state_last[b, :] = state
            state_batch[b, :, :] = state_all

        if len(self.placeholders) < 1:
            ph0 = PyTensor(self.name + "/h_state", state_batch, dtype=Dtype.FP32)
            ph1 = PyTensor(self.name + "/lut_in", lut_in_batch, dtype=Dtype.FP32)
            self.placeholders.append(ph0)
            self.placeholders.append(ph1)
        self.placeholders[0].betensor = state_batch
        self.placeholders[1].betensor = lut_in_batch

    else:
        scale = torch.tensor(self.params["scale_value"], device=inp0.betensor.device)
        shift = torch.tensor(self.params["shift_value"], device=inp0.betensor.device)
        lut_table = self.constants['lut'] if 'lut' in self.constants else None
        input_qbits = dtype2bits(self.inputs[0].dtype)
        if input_qbits <= 8:
            act_qmax = 2 ** 31 - 1
            act_qmin = -2 ** 31
        elif input_qbits <= 16:
            act_qmax = 2 ** 47 - 1
            act_qmin = -2 ** 47
        qmin, qmax = dtype2range(self.outputs[0].dtype)
        qbits = self.outputs[0].qbits
        for b in range(batch_size):
            state = torch.unsqueeze(in_state[b], dim=0).float()
            state_all = torch.zeros([0, cell_size], device=inp0.betensor.device)
            for ts in range(time_step):
                in_ts = torch.reshape(input_seq[b, ts, :], (-1, input_size)).float()
                #x*wx + h*wh + bias
                x_wx_sum = torch.matmul(in_ts, wx)
                h_wh_sum = torch.matmul(state, wh)
                re_scaled_x_wx_sum = linear_requantize(x_wx_sum, scale[0], shift[0], 0, act_qmin, act_qmax)
                re_scaled_h_wh_sum = linear_requantize(h_wh_sum, scale[1], shift[1], 0, act_qmin, act_qmax)
                xw_hw_sum = re_scaled_x_wx_sum + re_scaled_h_wh_sum + torch.unsqueeze(bias, 0)
                if activations in ['TANH', 'SIGMOID']:
                    xw_hw_sum = linear_requantize(xw_hw_sum, scale[2], shift[2], 0, qmin, qmax)
                    state = lookup_lut_powerof2(xw_hw_sum, lut_table.betensor, qbits, True,
                                                dtype2bits(lut_table.dtype), is_signed(lut_table.dtype))
                elif activations == "RELU":
                    xw_hw_sum = torch.nn.functional.relu(xw_hw_sum)
                    state = linear_requantize(xw_hw_sum, scale[2], shift[2], outp.zerop, outp.qmin, outp.qmax)
                elif activations in ["NONE"]:
                    state = linear_requantize(xw_hw_sum, scale[2], shift[2], outp.zerop, outp.qmin, outp.qmax)
                elif activations == "CLIP":
                    clip_max, clip_min = self.get_param('clip_max'), self.get_param('clip_min')
                    xw_hw_sum = torch.clamp(xw_hw_sum, clip_min, clip_max)
                    state = linear_requantize(
                        xw_hw_sum, scale[2], shift[2], 0, self.outputs[0].qmin, self.outputs[0].qmax)
                elif activations == "SIGN_BIT":
                    less_zero = xw_hw_sum < 0
                    larger_zero = xw_hw_sum >= 0
                    xw_hw_sum[less_zero] = 1
                    xw_hw_sum[larger_zero] = 0
                    state = xw_hw_sum
                state_all = torch.cat((state_all, state), dim=0)

            state_last[b, :] = state
            state_batch[b, :, :] = state_all

    if direction == "reverse":
        state_batch = torch.flip(state_batch, [1])

    self.outputs[0].betensor = state_batch
    self.outputs[1].betensor = state_last

    return (state_batch, state_last)


@quant_register(OpType.RNN)
def rnn_quantize(self, *args):
    q_mode_weight = self.attrs["q_mode_weight"]
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization of activations")
    q_bits_weight = self.attrs["q_bits_weight"]
    q_bits_bias = self.attrs["q_bits_bias"]
    q_bits_activation = self.attrs["q_bits_activation"]

    inp = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]
    [batch_size, time_step, input_size] = inp.betensor.shape

    dtype = dtype2str(bits2dtype(q_bits_activation, is_signed=True))
    out_sign = is_signed(inp1.dtype)

    activations = self.get_param('activations')
    cell_size = self.get_param('cell_size')
    weights = self.constants["weights"]
    w = weights.betensor.permute(1, 0)
    bias = self.constants["biases"]
    b = bias.betensor

    # quantize weights
    quantized_w = []
    w_scale = []
    w_zerop = []
    for idx, weights_name in enumerate(split_weights_name):
        w_scale_expand = []
        w_zerop_expand = []
        w_ele_t = self.constants[weights_name]
        w_ele_t.qbits = q_bits_weight
        w_ele_t.qinvariant = False
        w_ele = self.constants[weights_name].betensor
        w_ele_t.scale, w_ele_t.zerop, w_ele_t.qmin, w_ele_t.qmax, w_ele_t.dtype = \
            get_linear_quant_params_from_tensor(w_ele_t, q_mode_weight, w_ele_t.qbits, is_signed=True)

        w_scale_expand.extend([w_ele_t.scale] * w_ele.shape[0])
        w_zerop_expand.extend([w_ele_t.zerop] * w_ele.shape[0])
        scale_zp_shape = [w_ele.shape[0]] + [1 for i in range(len(w_ele.shape)-1)]
        w_scale.append(w_ele_t.scale)
        w_zerop.append(w_ele_t.zerop)
        w_scale_expand = torch.tensor(w_scale_expand, device=inp.betensor.device)
        w_zerop_expand = torch.tensor(w_zerop_expand, device=inp.betensor.device)
        quantized_ele = linear_quantize_clip(w_ele_t.betensor, w_scale_expand.reshape(scale_zp_shape),
                                             w_zerop_expand.reshape(scale_zp_shape), w_ele_t.qmin, w_ele_t.qmax)
        quantized_w.append(quantized_ele)
        self.constants.pop(weights_name)
    quantized_wx, quantized_wh = quantized_w
    wx_scale, wh_scale = w_scale

    quantized_weight = torch.cat((quantized_wx, quantized_wh), dim=1).contiguous()
    weights.scale = w_scale
    weights.zerop = w_zerop
    weights.betensor = quantized_weight
    weights.qbits = q_bits_weight
    weights.dtype = bits2dtype(weights.qbits, is_signed=True)
    weights.qinvariant = False

    h_all = self.placeholders[0]
    h_all.scale, h_all.zerop, h_all.qmin, h_all.qmax, h_all.dtype = \
        get_linear_quant_params_from_tensor(h_all, QuantMode.to_symmetric(
            q_mode_activation), q_bits_activation, is_signed=True)
    h_all.qbits = q_bits_activation
    h_all.qinvariant = False
    h_scale = h_all.scale

    outp0 = self.outputs[0]
    outp1 = self.outputs[1]
    outp0.scale, outp0.zerop, outp0.qmin, outp0.qmax, outp0.dtype = \
        h_all.scale, h_all.zerop, h_all.qmin, h_all.qmax, h_all.dtype
    outp0.qbits = q_bits_activation
    outp0.qinvariant = False
    outp1.scale, outp1.zerop, outp1.qmin, outp1.qmax, outp1.dtype = \
        h_all.scale, h_all.zerop, h_all.qmin, h_all.qmax, h_all.dtype
    outp1.qbits = q_bits_activation
    outp1.qinvariant = False

    xwx_scale = inp.scale * wx_scale
    hwh_scale = h_scale * wh_scale
    gb_scale = min(xwx_scale, hwh_scale)
    gb_zerop = 0
    qmin = -2**(q_bits_bias-1)
    qmax = 2**(q_bits_bias-1) - 1
    bias_q = linear_quantize_clip(b, gb_scale, gb_zerop, qmin, qmax)
    bias.zerop = 0
    bias.scale = gb_scale
    bias.qmin = qmin
    bias.qmax = qmax
    bias.qbits = q_bits_bias
    bias.betensor = bias_q
    bias.dtype = bits2dtype(bias.qbits, is_signed=True)
    bias.qinvariant = False

    absorb_input_h_zp_to_bias(self, *args)

    if activations in ["TANH", "SIGMOID"]:
        # save origin value
        _tensor_default_property = get_tensor_default_property()
        bak_inp_tensor_property = dict()
        t = self.inputs[0]
        property = {}
        for p in _tensor_default_property:
            property.update({p: t.__getattribute__(p)})
        bak_inp_tensor_property.update({t.name: property})
        bak_out_tensor_property = dict()
        t = self.outputs[0]
        property = {}
        for p in _tensor_default_property:
            property.update({p: t.__getattribute__(p)})
        bak_out_tensor_property.update({t.name: property})
        # self.force_dtype_int = True

        placeholder = self.placeholders[1]
        f_lut_in = placeholder.betensor
        lut_clamp_min, lut_clamp_max = g_rnn_activation_clamp[q_bits_activation][(self.type, activations)]
        f_lut_in = torch.clamp(f_lut_in, lut_clamp_min, lut_clamp_max)
        f_lut_out = g_rnn_activation_func[activations][1](f_lut_in)

        # placeholder.betensor = f_lut_in
        placeholder.max = f_lut_in.max()
        placeholder.min = f_lut_in.min()
        placeholder.scale, placeholder.zerop, placeholder.qmin, placeholder.qmax, placeholder.dtype = \
            get_linear_quant_params_from_tensor(placeholder, q_mode_activation, q_bits_activation, is_signed=True)
        placeholder.qbits = q_bits_activation
        placeholder.qinvariant = False

        ph2 = PyTensor(self.name + "/f_lut_out_clamp", f_lut_out, dtype=Dtype.FP32)
        ph2.max = f_lut_out.max()
        ph2.min = f_lut_out.min()
        lut = generate_activation_lut(self, activations, placeholder, ph2, *args)
        self.constants['lut'] = lut

        # get origin value
        for p in _tensor_default_property:
            self.inputs[0].__setattr__(p, bak_inp_tensor_property[self.inputs[0].name][p])
            self.outputs[0].__setattr__(p, bak_out_tensor_property[self.outputs[0].name][p])
        # self.force_dtype_int = False
        # self.inputs[0].betensor = bak_input0_betensor

        f_in_scale, f_out_scale = placeholder.scale, ph2.scale

        out_do_scale, out_do_scale_type, out_do_shift, out_do_shift_type = \
            get_scale_approximation_params(f_in_scale / gb_scale,
                                           q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)
    elif activations in ['RELU', "NONE"]:
        out_do_scale, out_do_scale_type, out_do_shift, out_do_shift_type = \
            get_scale_approximation_params(h_scale / gb_scale,
                                           q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)
    elif activations == "CLIP":
        clip_min = self.get_param('clip_min')
        clip_max = self.get_param('clip_max')

        self.params['clip_max'] = int(clip_max * gb_scale)
        self.params['clip_min'] = int(clip_min * gb_scale)

        out_do_scale, out_do_scale_type, out_do_shift, out_do_shift_type = \
            get_scale_approximation_params(outp0.scale / gb_scale, mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)

    elif activations == "SIGN_BIT":
        for out in self.outputs:
            out.scale = 1
            out.zerop = 0
            out.qbits = q_bits_activation
            out.qinvariant = True
            out.dtype = bits2dtype(out.qbits, out_sign)
            out.qmin, out.qmax = dtype2range(out.dtype)
        out_do_scale = 1
        out_do_scale_type = bits2dtype(q_bits_activation, False)
        out_do_shift = 0
        out_do_shift_type = bits2dtype(q_bits_activation, True)
    else:
        OPT_FATAL("layer_id=%s, type=%s, currently don't support activations(%s)" % (
            str(self.attrs['layer_id']), str(self.type), str(activations)))

    xwx_do_scale, xwx_do_scale_type, xwx_do_shift, xwx_do_shift_type = \
        get_scale_approximation_params(gb_scale / xwx_scale,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)
    hwh_do_scale, hwh_do_scale_type, hwh_do_shift, hwh_do_shift_type = \
        get_scale_approximation_params(gb_scale / hwh_scale,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)

    do_scale = torch.tensor([xwx_do_scale, hwh_do_scale, out_do_scale], device=inp.betensor.device)
    do_shift = torch.tensor([xwx_do_shift, hwh_do_shift, out_do_shift], device=inp.betensor.device)
    _, do_scale_type = range2dtype(0, do_scale.max().item())  # bits2dtype(q_bits_activation, is_signed=False)
    _, do_shift_type = range2dtype(do_shift.min().item(), do_shift.max().item(), force_int=True)
    scale_value = do_scale.cpu().numpy().astype(dtype2nptype(do_scale_type)).tolist()
    shift_value = do_shift.cpu().numpy().astype(dtype2nptype(do_shift_type)).tolist()
    self.params["shift_value"] = shift_value
    self.params["shift_type"] = [do_shift_type] * len(shift_value)
    self.params["scale_value"] = scale_value
    self.params["scale_type"] = [do_scale_type] * len(scale_value)
