# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.logger import OPT_ERROR
import torch


register_optype('AffineGrid')


def get_grid_vec(length, align_corners, dev):
    grid = torch.linspace(-1, 1, length, device=dev)
    if not align_corners:
        grid = grid * (length - 1) / length
    return grid


def generate_base_grid_2d(node, H, W):
    row = node.constants['w_lut'].betensor.double()
    col = node.constants['h_lut'].betensor.double()
    row_vec = row.unsqueeze(0)
    col_vec = col.unsqueeze(1)
    row_vec = row_vec.repeat(H, 1)
    col_vec = col_vec.repeat(1, W)
    base_grid = torch.stack((row_vec, col_vec), dim=2).reshape((H * W, 2)).permute(1, 0)
    return base_grid


def generate_base_grid_3d(node, D, H, W):
    row = node.constants['w_lut'].betensor.double()
    col = node.constants['h_lut'].betensor.double()
    slice = node.constants['d_lut'].betensor.double()
    row_vec = row.unsqueeze(0).unsqueeze(1)
    col_vec = col.unsqueeze(0).unsqueeze(2)
    slice_vec = slice.unsqueeze(1).unsqueeze(2)
    row_vec = row_vec.repeat(D, H, 1)
    col_vec = col_vec.repeat(D, 1, W)
    slice_vec = slice_vec.repeat(1, H, W)
    base_grid = torch.stack((row_vec, col_vec, slice_vec), dim=3).reshape((D * H * W, 3)).permute(1, 0)
    return base_grid


def affine_grid_generator_2d(node, theta, size):
    N, H, W, C, = size[0], size[1], size[2], size[3]
    output = torch.zeros([N, H, W, 2], device=theta.device)
    shift = node.get_ir_field(['shift_value'], default_value=0)
    scale = node.get_ir_field(['scale_value'], default_value=1)
    multiplier_bits = node.get_param('multiplier_bits', optional=True, default_value=0)
    qmin, qmax = bits2range(32, True)

    base_grid = generate_base_grid_2d(node, H, W)
    for b in range(N):
        theta_flatten = theta[b].flatten()
        theta_R = torch.tensor([
            [theta_flatten[0], theta_flatten[1]],
            [theta_flatten[3], theta_flatten[4]]
        ], device=theta.device).double()
        theta_T = torch.tensor([
            [theta_flatten[2]],
            [theta_flatten[5]]
        ], device=theta.device).double()
        batch_out = theta_R @ base_grid + theta_T * (2**(multiplier_bits))
        batch_out = torch.clamp(batch_out, qmin, qmax)
        batch_out = linear_requantize(batch_out, scale, shift,
                                      node.outputs[0].zerop, node.outputs[0].qmin, node.outputs[0].qmax)
        batch_out = batch_out.reshape(2, H, W).permute(1, 2, 0)
        output[b, ...] = batch_out
    return output


def affine_grid_generator_3d(node, theta, size):
    N, D, H, W, C, = size[0], size[1], size[2], size[3], size[4]
    output = torch.zeros([N, D, H, W, 3], device=theta.device)
    shift = node.get_ir_field(['shift_value'], default_value=0)
    scale = node.get_ir_field(['scale_value'], default_value=1)
    multiplier_bits = node.get_param('multiplier_bits', optional=True, default_value=0)
    qmin, qmax = bits2range(32, True)

    base_grid = generate_base_grid_3d(node, D, H, W)
    for b in range(N):
        theta_flatten = theta[b].flatten()
        theta_R = torch.tensor([
            [theta_flatten[0], theta_flatten[1], theta_flatten[2]],
            [theta_flatten[4], theta_flatten[5], theta_flatten[6]],
            [theta_flatten[8], theta_flatten[9], theta_flatten[10]]
        ], device=theta.device).double()
        theta_T = torch.tensor([
            [theta_flatten[3]],
            [theta_flatten[7]],
            [theta_flatten[11]]
        ], device=theta.device).double()
        batch_out = theta_R @ base_grid + theta_T * (2**(multiplier_bits))
        batch_out = torch.clamp(batch_out, qmin, qmax)
        batch_out = linear_requantize(batch_out, scale, shift, node.outputs[0].zerop, node.outputs[0].qmin,
                                      node.outputs[0].qmax)
        batch_out = batch_out.reshape(3, D, H, W).permute(1, 2, 3, 0)
        output[b, ...] = batch_out
    return output


@op_register(OpType.AffineGrid)
def affinegrid(self, *args):
    theta = self.inputs[0].betensor
    # size = self.inputs[1].betensor

    align_corners = self.get_param('align_corners')

    if theta.dim() != 3:
        OPT_ERROR(
            f"GripSaAffineGridmple op now only supports 3-dims, now input0 dim is {str(theta.dim())}.")

    size = list(self.outputs[0].ir_shape)
    is_5dims = True if len(size) == 5 else False
    if self.quantized:
        if is_5dims:
            out = affine_grid_generator_3d(self, theta, size)
        else:
            out = affine_grid_generator_2d(self, theta, size)
    else:
        if is_5dims:
            size[0], size[1], size[2], size[3], size[4] = size[0], size[4], size[1], size[2], size[3]
        else:
            size[0], size[1], size[2], size[3] = size[0], size[3], size[1], size[2]
        out = torch.nn.functional.affine_grid(theta, torch.Size(size), align_corners=align_corners)
    self.outputs[0].betensor = out
    return out


@quant_register(OpType.AffineGrid)
def affinegrid_quantize(self, *args):
    import math
    q_mode_activation = self.attrs["q_mode_activation"]
    q_bits_activation = self.attrs["q_bits_activation"]
    align_corners = self.get_param('align_corners')

    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out = self.outputs[0]
    dev = inp0.betensor.device
    out_signed = True
    out.qbits = q_bits_activation
    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, out.qbits, is_signed=out_signed)
    out.qinvariant = False

    size = list(self.outputs[0].ir_shape)
    is_5dims = True if len(size) == 5 else False
    if is_5dims:
        _, D, H, W, _, = size[0], size[1], size[2], size[3], size[4]
        max_length = max(D, H, W)
    else:
        _, H, W, _ = size[0], size[1], size[2], size[3]
        max_length = max(H, W)

    multiplier_bits = max(8, min(14, int(math.ceil(math.log2(max_length)))+1))
    overlap_bits, _ = range2bits(-(2**multiplier_bits), 2**multiplier_bits, force_int=True)
    grid_dtype = bits2dtype(overlap_bits, True)
    grid_min, grid_max = dtype2range(grid_dtype)

    W_lut = get_grid_vec(W, align_corners, dev)
    H_lut = get_grid_vec(H, align_corners, dev)
    W_lut = torch.clamp((W_lut * (2**multiplier_bits)).round(), grid_min, grid_max)
    H_lut = torch.clamp((H_lut * (2**multiplier_bits)).round(), grid_min, grid_max)
    self.constants["w_lut"] = PyTensor("W_lut", W_lut, dtype=grid_dtype)
    self.constants["w_lut"].dtype = grid_dtype
    self.constants["h_lut"] = PyTensor("H_lut", H_lut, dtype=grid_dtype)
    self.constants["h_lut"].dtype = grid_dtype
    if is_5dims:
        D_lut = get_grid_vec(D, align_corners, dev)
        D_lut = torch.clamp((D_lut * (2**multiplier_bits)).round(), grid_min, grid_max)
        self.constants["d_lut"] = PyTensor("D_lut", D_lut, dtype=grid_dtype)
        self.constants["d_lut"].dtype = grid_dtype

    ts = out.scale / (inp0.scale * (2**(multiplier_bits)))
    do_scale, do_scale_type, do_shift, do_shift_type = \
        get_scale_approximation_params(ts,
                                       q_bits_activation,
                                       force_shift_positive=self.force_shift_positive)

    doscale_name = 'scale' if is_torch_tensor_with_multi_data(do_scale) else 'scale_value'
    doshift_name = 'shift' if is_torch_tensor_with_multi_data(do_shift) else 'shift_value'
    self.set_ir_field(doscale_name, do_scale, do_scale_type)
    self.set_ir_field(doshift_name, do_shift, do_shift_type)
    if not is_torch_tensor_with_multi_data(do_scale):
        self.params["shift_type"] = do_shift_type
        self.params["scale_type"] = do_scale_type

    self.params['multiplier_bits'] = multiplier_bits
