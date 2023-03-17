# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import tensorflow.compat.v1 as tf
# tf.enable_eager_execution()


def conv2d_tf_impl(self, *args):
    def _impl(inp, weight, bias, padding, stride, dilation, pad_val):
        inp = tf.convert_to_tensor(inp, dtype=tf.float32)
        weight = tf.convert_to_tensor(weight, dtype=tf.float32)
        bias = tf.convert_to_tensor(bias, dtype=tf.float32)

        padded_inp = tf.pad(inp, padding, mode='CONSTANT', constant_values=pad_val)
        weight = tf.transpose(weight, perm=[1, 2, 3, 0])

        conv = tf.nn.conv2d(padded_inp,
                            filters=weight,
                            strides=stride,
                            padding='VALID',
                            data_format='NHWC',
                            dilations=dilation)
        conv = tf.nn.bias_add(conv, bias)
        if not tf.executing_eagerly():
            with tf.Session().as_default():
                outt = conv.eval()
        else:
            outt = conv.numpy()
        return outt

    inp = self.inputs[0].betensor
    device = self.inputs[0].betensor.device
    weights = self.constants['weights'].betensor.clone()
    bias = self.constants['biases'].betensor.clone()
    stride = [self.get_param("stride_y"), self.get_param("stride_x")]
    dilation = [self.get_param('dilation_y'), self.get_param('dilation_x')]
    padding = [[0, 0],
               [self.get_param('pad_top'), self.get_param('pad_bottom')],
               [self.get_param('pad_left'), self.get_param('pad_right')],
               [0, 0]]
    pad_val = 0
    if self.quantized:
        # input's zerop has been absorbed to bias.
        # inp += self.inputs[0].zerop
        pad_val = -self.inputs[0].zerop
        w_zp = self.constants["weights"].zerop
        w_zshape = [1] * weights.dim()
        w_zshape[0] = -1
        weights += w_zp.reshape(w_zshape) if isinstance(w_zp, torch.Tensor) else w_zp
        bias += self.constants['biases'].zerop

    inp = inp.cpu().numpy()
    weight = weights.cpu().numpy()
    bias = bias.cpu().numpy()
    x = _impl(inp, weight, bias, padding, stride, dilation, pad_val)
    x = torch.from_numpy(x).to(device)

    return x
