# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.ops.lpnormalization import isqrt_forward
import torch

register_optype('EmbeddingLookupSparse')


def segment_sum(data, indices):
    idx = indices[..., 0]
    index_range = (idx[-1] + 1).item()
    zero_dim = [1] + [data.shape[ax] for ax in range(data.dim())[1:]]
    output = None
    for j in range(index_range):
        if j in idx:
            j_index = torch.eq(idx, j)
            j_index_range = torch.nonzero(j_index)
            tmp_data = torch.index_select(data, dim=0, index=torch.squeeze(j_index_range))
            tmp_data = torch.sum(tmp_data, dim=0, keepdim=True)
        else:
            tmp_data = torch.full(zero_dim, 0, device=data.device)
        output = tmp_data if output == None else torch.cat((output, tmp_data), dim=0)
    return output


@quant_register(OpType.EmbeddingLookupSparse)
def lookupsparse_quantize(self, *args):
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("Currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]
    combiner = self.get_param('combiner').lower()
    max_norm = self.get_param('max_norm', optional=True, default_value="NONE").lower()

    if max_norm != "none":
        # calculate norm scale
        OPT_FATAL("only support max_norm is NONE, but now max_norm is: %s for EmbeddingLookupSparse in node:%s" %
                  (max_norm, self.name))
    inp0_signed = (self.inputs[0].qmin + self.inputs[0].zerop) < 0
    inp3_signed = (self.inputs[3].qmin + self.inputs[3].zerop) < 0

    # calculate segment(params) scale
    params_mul_weight = self.placeholders[0]
    params_mul_weight.qbits = q_bits_activation
    params_mul_weight.qinvariant = False
    params_weight_signed = inp0_signed or inp3_signed
    params_mul_weight.scale, params_mul_weight.zerop, params_mul_weight.qmin, params_mul_weight.qmax, params_mul_weight.dtype = get_linear_quant_params_from_tensor(
        params_mul_weight, QuantMode.to_symmetric(q_mode_activation), params_mul_weight.qbits, params_weight_signed)

    # calculate segment(weight) scale
    weights_sum = self.placeholders[1]
    weights_sum.qbits = q_bits_activation
    weights_sum.qinvariant = False
    weights_sum_signed = inp3_signed
    weights_sum.scale, weights_sum.zerop, weights_sum.qmin, weights_sum.qmax, weights_sum.dtype = get_linear_quant_params_from_tensor(
        weights_sum, QuantMode.to_symmetric(q_mode_activation), weights_sum.qbits, weights_sum_signed)

    outputs = self.outputs[0]
    outputs.qbits = q_bits_activation
    outputs.qinvariant = False
    outputs_signed = inp0_signed or inp3_signed or self.force_dtype_int
    outputs.scale, outputs.zerop, outputs.qmin, outputs.qmax, outputs.dtype = get_linear_quant_params_from_tensor(
        outputs, q_mode_activation, outputs.qbits, outputs_signed)

    if combiner == 'sum':
        out_scale, out_scale_type, out_shift, out_shift_type = \
            get_scale_approximation_params(outputs.scale / (self.inputs[0].scale * self.inputs[3].scale), mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)
        self.params["shift_value"] = int(out_shift)
        self.params["shift_type"] = out_shift_type
        self.params["scale_value"] = int(out_scale)
        self.params["scale_type"] = out_scale_type

    elif combiner == 'mean':
        out_do_scale, out_do_scale_type, out_do_shift, out_do_shift_type = \
            get_scale_approximation_params(outputs.scale / (self.inputs[0].scale),
                                           mult_bits=16,
                                           force_shift_positive=self.force_shift_positive)

        self.params["shift_value"] = int(out_do_shift)
        self.params["shift_type"] = out_do_shift_type
        self.params["scale_value"] = int(out_do_scale)
        self.params["scale_type"] = out_do_scale_type
    elif combiner == 'sqrtn':
        local_scale = params_mul_weight.scale / (self.inputs[0].scale * self.inputs[3].scale)
        params_scale, params_scale_type, params_shift, params_shift_type = \
            get_scale_approximation_params(local_scale, mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)

        do_scale, do_scale_type, do_shift, do_shift_type = \
            get_scale_approximation_params(outputs.scale * self.inputs[3].scale / params_mul_weight.scale,
                                           mult_bits=q_bits_activation,
                                           force_shift_positive=self.force_shift_positive)
        pre_shift = 0
        scale_bits = torch.ceil(torch.log2(torch.tensor(do_scale))).item()
        input_bits = params_mul_weight.qbits + (0 if is_signed(params_mul_weight.dtype) else 1)
        if (input_bits + scale_bits) > 16:
            pre_shift = (input_bits + scale_bits) - 16
        self.params['reciprocal_shift_value'] = int(pre_shift)
        self.params['reciprocal_shift_type'] = Dtype.INT8
        self.params["shift_value"] = [int(params_shift), int(do_shift)]
        self.params["shift_type"] = [params_shift_type, do_shift_type]
        self.params["scale_value"] = [int(params_scale), int(do_scale)]
        self.params["scale_type"] = [params_scale_type, do_scale_type]
        self.constants["lut"] = PyTensor(
            self.name+"/isqrt_lut", torch.tensor(inverse_sqrt_table).cpu().numpy().astype(dtype2nptype(Dtype.INT16)))
        self.constants["lut"].dtype = Dtype.INT16
    else:
        OPT_FATAL("unsupported method: %s for EmbeddingLookupSparse in node:%s" % (combiner, self.name))


@op_register(OpType.EmbeddingLookupSparse)
def lookupsparse(self, *args):
    params = self.inputs[0]
    indices = self.inputs[1].betensor.long()
    sp_ids_value = self.inputs[2].betensor.squeeze().long()
    weights_value = self.inputs[3].betensor.squeeze()
    out = self.outputs[0]

    combiner = self.get_param('combiner').lower()
    max_norm = self.get_param('max_norm', optional=True, default_value="NONE").lower()
    if max_norm != "none":
        OPT_FATAL("only support max_norm is NONE, but now max_norm is: %s for EmbeddingLookupSparse in node:%s" %
                  (max_norm, self.name))
    if combiner not in ["sum", "mean", "sqrtn"]:
        OPT_FATAL("unsupported method: %s for EmbeddingLookupSparse in node:%s" % (combiner, self.name))

    input_dim = params.betensor.dim()
    norm_dim = [ax for ax in range(1, input_dim)]
    weights_shape = [-1] + [1 for ax in range(1, input_dim)]
    if indices.ndim == 1:
        indices = torch.unsqueeze(indices, dim=-1)

    # check invalid
    invalid_mask = torch.bitwise_or(sp_ids_value < 0, sp_ids_value >= params.betensor.shape[0])
    if True in invalid_mask:
        OPT_FATAL("the 2th input value of EmbeddingLookupSparse in node %s must be [0, %d)" % (
            self.name, params.betensor.shape[0]))
    sorted_indices = torch.sort(indices[..., 0], dim=-1)[0].clone()
    diff = torch.abs(sorted_indices - indices[..., 0])
    if diff.max():
        OPT_FATAL(
            "the 1th input indices[...,0] must be sorted and increasing of EmbeddingLookupSparse in node %s " % (self.name))

    params_data = torch.index_select(params.betensor, dim=0, index=sp_ids_value)
    weights_value = torch.reshape(weights_value, weights_shape)
    if not self.quantized:
        normal_parmas = params_data
        if max_norm is not None:
            pass
            # max_norm = float(max_norm)
            # norm = torch.sqrt(torch.sum(params_data*params_data, dim = norm_dim, keepdim=True)).squeeze()
            # for idx in range(params_data.shape[0]):
            #     params_data[idx] = (params_data[idx] * max_norm) / max(norm[idx], max_norm)
            # normal_parmas = params_data
        params_weight = normal_parmas * weights_value
        params_sum = segment_sum(params_weight, indices)
        if combiner == 'sum':
            weights_sum = torch.zeros_like(weights_value, device=weights_value.device)
            out.betensor = params_sum
            pass
        elif combiner == 'mean':
            weights_sum = segment_sum(weights_value, indices)
            output = params_sum / weights_sum
            zeros_tensor = torch.zeros_like(output, device=params_sum.device)
            out.betensor = torch.where(torch.bitwise_or(torch.isinf(output), torch.isnan(output)), zeros_tensor, output)
        else:  # combiner == 'sqrtn':
            weights_squared = weights_value * weights_value
            weights_sum = segment_sum(weights_squared, indices)
            weights_sqrt = torch.sqrt(weights_sum)
            output = params_sum / weights_sqrt
            zeros_tensor = torch.zeros_like(output, device=params_sum.device)
            out.betensor = torch.where(torch.bitwise_or(torch.isinf(output), torch.isnan(output)), zeros_tensor, output)

        if len(self.placeholders) < 1:
            ph0 = PyTensor(self.name+"/params_sum", params_sum.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            ph1 = PyTensor(self.name+"/weights_sum", weights_sum.cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
            self.placeholders.append(ph0)
            self.placeholders.append(ph1)
        self.placeholders[0].betensor = params_sum
        self.placeholders[1].betensor = weights_sum
    else:
        param_signed = (self.inputs[0].qmin + self.inputs[0].zerop) < 0
        weight_signed = (self.inputs[3].qmin + self.inputs[3].zerop) < 0

        do_scale = self.get_param('scale_value')
        do_shift = self.get_param('shift_value')

        params_data = (params_data + self.inputs[0].zerop).long()
        weights_value = (weights_value + self.inputs[3].zerop).long()
        params_weight = (params_data * weights_value).long()
        params_sum = segment_sum(params_weight, indices).long()
        if combiner == 'sum':
            out.betensor = linear_requantize(params_sum, do_scale, do_shift, out.zerop, out.qmin, out.qmax)
        elif combiner == 'mean':
            q16_min, q16_max = bits2range(16, weight_signed)
            act_qmin, act_qmax = bits2range(32, param_signed)

            weights_sum = segment_sum(weights_value, indices)
            params_sum = params_sum * do_scale
            params_sum_bits, _ = range2bits(params_sum.min(), params_sum.max(), force_int=param_signed)
            weights_sum_bits, _ = range2bits(weights_sum.min(), weights_sum.max(), force_int=weight_signed)
            preshift0 = (params_sum_bits - 32) if params_sum_bits > 32 else 0
            preshift1 = (weights_sum_bits - 16) if weights_sum_bits > 16 else 0
            params_sum = torch.clamp(params_sum >> preshift0, act_qmin, act_qmax)
            weights_sum = torch.clamp(weights_sum >> preshift1, q16_min, q16_max)
            repeat_size = [params_sum.shape[ax] // weights_sum.shape[ax] for ax in range(params_sum.dim())]
            weights_sum = weights_sum.repeat(repeat_size)
            zeros_tensor = torch.zeros_like(params_sum, device=params_sum.device).int()
            output = torch.where(weights_sum != 0, torch.trunc(params_sum.double()/weights_sum).int(), zeros_tensor)
            out.betensor = linear_requantize(output, 1, do_shift-preshift0+preshift1,
                                             out.zerop, out.qmin, out.qmax).int()
        else:
            pre_shift = self.get_param('reciprocal_shift_value')
            q16_min, q16_max = bits2range(16, True)
            act_qmin, act_qmax = bits2range(32, True)
            params_qmin, params_qmax = bits2range(self.inputs[0].qbits, param_signed)

            x = linear_requantize(params_sum, do_scale[0], do_shift[0], 0, params_qmin, params_qmax)
            weights_squared = weights_value * weights_value
            psum = segment_sum(weights_squared, indices)
            rsqnorm, rsqnorm_shift = isqrt_forward(psum)
            x = torch.clamp((x * do_scale[1]), act_qmin, act_qmax)
            x = torch.clamp(x >> pre_shift, q16_min, q16_max)
            x = torch.clamp((x * rsqnorm), act_qmin, act_qmax)
            out.betensor = linear_requantize(x, 1, do_shift[1] + 31 -
                                             pre_shift - rsqnorm_shift, out.zerop, out.qmin, out.qmax)

    return self.outputs[0].betensor
