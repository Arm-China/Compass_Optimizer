# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.ops.conv import *
from AIPUBuilder.Optimizer.ops.activation import apply_with_activation
import torch


def checkoutInputValid(node, input, image_dims, kernel_dims, output_shape):
    if input.ndim != 3:
        OPT_ERROR(f"{node}, currently input dim only support 3 dims.")
        return False
    if image_dims.numel() < 2 or image_dims.numel() > 3:
        OPT_ERROR(f"{node}, currently image size(2nd input) only support 2 or 3.")
        return False
    if image_dims.numel() != kernel_dims.numel():
        OPT_ERROR(f"{node}, currently image size(2nd input) must be equal to kernel size(3rd input).")
        return False
    if len(output_shape) != (image_dims.numel() + 2):
        OPT_ERROR(
            f"{node}, currently output dims must be equal to image size(2rd input) + 2, such as [N,C,*image_shape].")
        return False
    return True


def unfoldNd(node, input, kernel_size, kernel_size_numel):
    batch_size, in_channels = input.shape[0], input.shape[1]
    stride = (node.get_param("stride_y"), node.get_param("stride_x"))
    dilation = (node.get_param('dilation_y'), node.get_param('dilation_x'))
    padding = (node.get_param('pad_left'), node.get_param('pad_right'),
               node.get_param('pad_top'), node.get_param('pad_bottom'))
    repeat = [in_channels, 1] + [1 for _ in kernel_size]
    weight = torch.eye(kernel_size_numel, device=input.device, dtype=input.dtype).reshape(
        (kernel_size_numel, 1, *kernel_size)).repeat(*repeat)
    input = torch.nn.functional.pad(input, padding, mode='constant', value=-32767)
    unfold = torch.nn.functional.conv2d(
        input,
        weight,
        bias=None,
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=in_channels,
    )

    return unfold.reshape(batch_size, in_channels * kernel_size_numel, -1)


def col2im2D(node, inp, image_dims, kernel_dims):
    batch_size = inp.shape[0]
    in_channels_kernel_size_numel = inp.shape[1]
    kernel_size_numel = torch.prod(kernel_dims)
    output_size_numel = torch.prod(image_dims)
    in_channels = in_channels_kernel_size_numel // kernel_size_numel

    idx = torch.arange(output_size_numel, dtype=torch.float32, device=inp.device).reshape(1, 1, *image_dims)
    idx = unfoldNd(node, idx, kernel_dims, kernel_size_numel)

    inp = inp.reshape(batch_size, in_channels, -1)
    idx = idx.reshape(1, 1, -1).long().expand(batch_size, in_channels, -1)

    mask = idx == -32767
    inp[mask] = 0
    idx[mask] = 0

    out = torch.zeros(
        batch_size, in_channels, output_size_numel, dtype=inp.dtype, device=inp.device)

    out.scatter_add_(2, idx, inp)

    out = out.reshape(batch_size, in_channels, *image_dims)

    if node.quantized:
        do_scale = node.get_param('scale_value')
        do_shift = node.get_param('shift_value')
        out = linear_requantize(out, do_scale, do_shift,
                                node.outputs[0].zerop, node.outputs[0].qmin, node.outputs[0].qmax)

    return out


def col2imND(node, inp, image_dims, kernel_dims):
    from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
    padding = [node.get_param('pad_z_begin'), node.get_param('pad_y_begin'), node.get_param('pad_x_begin'),
               node.get_param('pad_z_end'), node.get_param('pad_y_end'), node.get_param('pad_x_end')]
    dilations = [node.get_param('dilation_z'), node.get_param('dilation_y'), node.get_param('dilation_x')]
    strides = [node.get_param('stride_z'), node.get_param('stride_y'), node.get_param('stride_x')]
    rank = image_dims.numel()
    image_shape_size = torch.prod(image_dims)
    kernel_shape_size = torch.prod(kernel_dims)
    adjusted_kernel_shape = torch_tensor(dilations, device=inp.device) * (kernel_dims - 1) + 1

    N = inp.shape[0]
    C = int(inp.shape[1] / kernel_shape_size)
    col_stride = C * image_shape_size  # 3*14*15*16
    col_data_stride = inp.shape[1] * inp.shape[2]
    batched_image_shape_dims = torch.cat(
        (torch_tensor([N, C], device=inp.device), torch.zeros([rank], device=inp.device)))
    batched_image_shape_dims[2:] = image_dims
    adjusted_image_shape_dims = image_dims - adjusted_kernel_shape + 1
    batched_image_shape_dims = batched_image_shape_dims.long().tolist()

    inp_flatten = inp.flatten()
    image_data = torch.zeros(batched_image_shape_dims, device=inp.device).flatten()
    for b in range(N):
        data_im = inp_flatten[b * col_data_stride:]
        im_shape = image_dims
        output_shape = adjusted_image_shape_dims
        channels_col = kernel_shape_size * C
        kernel_shape = adjusted_kernel_shape
        data_col = image_data[b * col_stride:]
        kernel_size = torch.prod(kernel_shape)
        d_offset = [0] * rank
        d_iter = [0] * rank
        for c_col in range(channels_col):
            offset = c_col
            for di in range(rank - 1, -1, -1):
                if di < (rank - 1):
                    offset = offset // kernel_shape[di + 1]
                d_offset[di] = int(offset) % kernel_shape[di]
            while True:
                index_col = c_col
                index_im = c_col // kernel_size
                is_padding = False
                for d_i in range(rank):
                    d = d_iter[d_i]
                    d_im = d * strides[d_i] - padding[d_i] + d_offset[d_i] * dilations[d_i]
                    is_padding = is_padding or (not (d_im >= 0 and d_im < im_shape[d_i]))
                    index_col *= output_shape[d_i]
                    index_col += d
                    index_im *= im_shape[d_i]
                    index_im += d_im
                if not is_padding:
                    if index_col < data_im.shape[0] and index_im < data_col.shape[0]:
                        data_col[index_im] += data_im[index_col]

                def NextPosition(shape, d_iter):
                    has_next_output = False
                    for d_x in range(rank - 1, -1, -1):
                        d_max = shape[d_x]
                        if d_iter[d_x] == (d_max - 1):
                            d_iter[d_x] = 0
                        else:
                            d_iter[d_x] += 1
                            has_next_output = True
                            break
                    return has_next_output, d_iter

                has_next_output, d_iter = NextPosition(output_shape, d_iter)
                if not has_next_output:
                    break

    output = torch.reshape(image_data, batched_image_shape_dims)

    if node.quantized:
        do_scale = node.get_param('scale_value')
        do_shift = node.get_param('shift_value')
        output = linear_requantize(output, do_scale, do_shift,
                                   node.outputs[0].zerop, node.outputs[0].qmin, node.outputs[0].qmax)

    return output


register_optype('Col2Im')


@op_register(OpType.Col2Im)
def col2im(self, *args):
    inp = self.inputs[0].betensor
    image_dims = self.inputs[1].betensor
    kernel_dims = self.inputs[2].betensor

    inputValid = checkoutInputValid(self, inp, image_dims, kernel_dims, self.outputs[0].ir_shape)
    if not inputValid:
        OPT_FATAL(f"{self}, input or output shape mismatch IR definition, please check! ")

    rank = image_dims.numel()
    if rank == 2:
        self.outputs[0].betensor = col2im2D(self, inp, image_dims, kernel_dims)
    else:
        self.outputs[0].betensor = col2imND(self, inp, image_dims, kernel_dims)

    return self.outputs[0].betensor


@quant_register(OpType.Col2Im)
def col2im_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    q_bits_activation = self.attrs["q_bits_activation"]
    q_mode_activation = self.attrs["q_mode_activation"]

    out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
        out, q_mode_activation, q_bits_activation, is_signed=is_signed(inp.dtype))
    out.qbits = q_bits_activation
    out.qinvariant = False

    local_rescale = out.scale / inp.scale
    do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(local_rescale,
                                                                                      mult_bits=q_bits_activation,
                                                                                      force_shift_positive=self.force_shift_positive)
    self.params["shift_value"] = int(do_shift)
    self.params["shift_type"] = do_shift_type
    self.params["scale_value"] = int(do_scale)
    self.params["scale_type"] = do_scale_type
