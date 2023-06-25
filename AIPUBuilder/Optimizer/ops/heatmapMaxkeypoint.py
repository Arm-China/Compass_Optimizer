# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch


register_optype('HeatmapMaxKeypoint')


@quant_register(OpType.HeatmapMaxKeypoint)
def HeatmapMaxKeypoint_quantized(self, *args):
    inp0, inp1 = self.inputs[0], self.inputs[1]
    out0, out1, out2 = self.outputs[0], self.outputs[1], self.outputs[2]
    q_mode_activation = self.attrs["q_mode_activation"]
    if QuantMode.is_per_channel(q_mode_activation) == True:
        OPT_FATAL("SufficientStatistics currently not support per-channel quantization")
    q_bits_activation = self.attrs["q_bits_activation"]

    out_signed_list = [is_signed(inp0.dtype) or self.force_dtype_int,
                       is_signed(inp1.dtype) or self.force_dtype_int,
                       False or self.force_dtype_int]
    out_bits_list = [dtype2bits(inp0.dtype), dtype2bits(inp1.dtype), dtype2bits(inp0.dtype)]
    for idx, _ in enumerate(self.outputs):
        out = self.outputs[idx]
        out.qinvariant = False
        out.qbits = out_bits_list[idx]
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, out.qbits, is_signed=out_signed_list[idx])

    heatmapSize = inp0.ir_shape[1]
    score_scale, score_scale_type, score_shift, score_shift_type = \
        get_scale_approximation_params(out0.scale / inp0.scale, mult_bits=16,
                                       force_shift_positive=self.force_shift_positive)
    box_scale, box_scale_type, box_shift, box_shift_type = \
        get_scale_approximation_params(out1.scale / inp1.scale / heatmapSize, mult_bits=16,
                                       force_shift_positive=self.force_shift_positive)

    self.params["score_shift_value"] = int(score_shift)
    self.params["score_shift_type"] = score_shift_type
    self.params["score_scale_value"] = int(score_scale)
    self.params["score_scale_type"] = score_scale_type

    self.params["box_shift_value"] = int(box_shift)
    self.params["box_shift_type"] = box_shift_type
    self.params["box_scale_value"] = int(box_scale)
    self.params["box_scale_type"] = box_scale_type


def getLocalGrid(box_score, maxIndexWidth, maxIndexHeight, numBoxes, heatmapSize, numKeypoints):
    localGrid = torch.zeros([numBoxes, numKeypoints, 3, 3], device=box_score.device)
    hmin = wmin = torch.ones_like(maxIndexWidth, device=maxIndexWidth.device)
    hmax = torch.ones_like(maxIndexHeight, device=maxIndexWidth.device) * (heatmapSize - 2)
    wmax = torch.ones_like(maxIndexWidth, device=maxIndexWidth.device) * (heatmapSize - 2)
    for dh in range(-1, 2):
        for dw in range(-1, 2):
            h = maxIndexHeight + dh
            w = maxIndexWidth + dw
            # use mirroring for out of bound indexing
            h = torch.where(h >= heatmapSize, hmax, h)
            h = torch.where(h < 0, hmin, h)

            w = torch.where(w >= heatmapSize, wmax, w)
            w = torch.where(w < 0, wmin, w)

            heatmapIndex = h * heatmapSize + w
            localGrid[:, :, dh+1, dw+1] = torch.gather(box_score, dim=-1, index=heatmapIndex)[..., 0]
    return localGrid


def solveForDelta(grid, numBoxes, numKeypoints, max_score):
    fpAtol = fpRtol = 1e-5
    delta_pair = torch.zeros([numBoxes, numKeypoints, 2], device=grid.device).float()
    deltaScore = max_score.clone().float()
    b0 = -(grid[:, :, 1, 2] - grid[:, :, 1, 0]) / 2.0
    b1 = -(grid[:, :, 2, 1] - grid[:, :, 0, 1]) / 2.0
    A00 = grid[:, :, 1, 0] - 2.0 * grid[:, :, 1, 1] + grid[:, :, 1, 2]
    A01 = (grid[:, :, 2, 2] - grid[:, :, 2, 0] - grid[:, :, 0, 2] + grid[:, :, 0, 0]) / 4.0
    A10 = A01
    A11 = grid[:, :, 0, 1] - 2.0 * grid[:, :, 1, 1] + grid[:, :, 2, 1]

    crossProd1 = A00 * A11
    crossProd2 = A01 * A10
    detA = crossProd1 - crossProd2
    invalid_mask = torch.abs(detA) < (fpAtol + fpRtol * crossProd1)
    valid_mask = ~invalid_mask

    detA_valid = detA[valid_mask]
    b0_valid = b0[valid_mask]
    b1_valid = b1[valid_mask]
    A00_valid = A00[valid_mask]
    A01_valid = A01[valid_mask]
    A10_valid = A10[valid_mask]
    A11_valid = A11[valid_mask]
    delta0 = (A11_valid * b0_valid - A01_valid * b1_valid) / detA_valid
    delta1 = (A00_valid * b1_valid - A10_valid * b0_valid) / detA_valid

    gt_thres_mask = torch.bitwise_or(torch.abs(delta0) > 1.5, torch.abs(delta1) > 1.5)
    if len(gt_thres_mask) > 0:
        scale = 1.5 / torch.maximum(torch.abs(delta0[gt_thres_mask]), torch.abs(delta1[gt_thres_mask]))
        delta0[gt_thres_mask] *= scale
        delta1[gt_thres_mask] *= scale

    valid_mask_flatten = valid_mask.flatten()
    delta_pair_flatten = delta_pair.reshape(numBoxes * numKeypoints, 2)
    deltaScore_flatten = deltaScore.reshape(numBoxes * numKeypoints)
    delta_pair_flatten[valid_mask_flatten, 0] = delta0
    delta_pair_flatten[valid_mask_flatten, 1] = delta1
    deltaScore_flatten[valid_mask_flatten] = grid[:, :, 1, 1][valid_mask] - b0_valid * delta0 - b1_valid * delta1 + \
        ((A00_valid * delta0 + A01_valid * delta1) * delta0 +
         (A10_valid * delta0 + A11_valid * delta1) * delta1) / 2.0

    delta_pair = delta_pair_flatten.reshape(delta_pair.shape)
    deltaScore = deltaScore_flatten.reshape(deltaScore.shape)

    return delta_pair, deltaScore


def solveForDelta_quant(localGrid, numBoxes, numKeypoints, heatmapsize_max_score):
    delta_pair = torch.zeros([numBoxes, numKeypoints, 2], device=localGrid.device).int()
    deltaScore = heatmapsize_max_score.clone().int()
    # multiply by 4 times
    b0 = -2 * (localGrid[:, :, 1, 2] - localGrid[:, :, 1, 0]).int()
    b1 = -2 * (localGrid[:, :, 2, 1] - localGrid[:, :, 0, 1]).int()
    A00 = 4 * (localGrid[:, :, 1, 0] - 2.0 * localGrid[:, :, 1, 1] + localGrid[:, :, 1, 2]).int()
    A01 = (localGrid[:, :, 2, 2] - localGrid[:, :, 2, 0] - localGrid[:, :, 0, 2] + localGrid[:, :, 0, 0]).int()
    A10 = A01
    A11 = 4 * (localGrid[:, :, 0, 1] - 2.0 * localGrid[:, :, 1, 1] + localGrid[:, :, 2, 1]).int()

    crossProd1 = A00 * A11
    crossProd2 = A01 * A10
    detA = crossProd1 - crossProd2
    invalid_mask = torch.abs(detA) == 0
    valid_mask = ~invalid_mask

    detA_valid = detA[valid_mask]
    b0_valid = b0[valid_mask]
    b1_valid = b1[valid_mask]
    A00_valid = A00[valid_mask]
    A01_valid = A01[valid_mask]
    A10_valid = A10[valid_mask]
    A11_valid = A11[valid_mask]
    delta0 = torch.div((A11_valid * b0_valid - A01_valid * b1_valid) * (2**8), detA_valid, rounding_mode='floor').int()
    delta1 = torch.div((A00_valid * b1_valid - A10_valid * b0_valid) * (2**8), detA_valid, rounding_mode='floor').int()

    gt_thres_mask = torch.bitwise_or(torch.abs(delta0) > (1.5 * 256), torch.abs(delta1) > (1.5 * 256))
    if len(gt_thres_mask) > 0:
        scale = torch.div(
            (1.5*256)*256, torch.maximum(torch.abs(delta0[gt_thres_mask]), torch.abs(delta1[gt_thres_mask])), rounding_mode='floor').long()
        delta0[gt_thres_mask] *= scale
        delta1[gt_thres_mask] *= scale
        delta0[gt_thres_mask] = delta0[gt_thres_mask].int() >> 8
        delta1[gt_thres_mask] = delta1[gt_thres_mask].int() >> 8

    s0 = localGrid[:, :, 1, 1][valid_mask] * 2**16
    s1 = b0_valid * delta0 * (2 ** 6)
    s2 = b1_valid * delta1 * (2 ** 6)
    s3 = (((A00_valid * delta0 + A01_valid * delta1) * delta0 + (A10_valid * delta0 + A11_valid * delta1) * delta1)) >> 3
    s = (s0-s1-s2+s3).int()

    valid_mask_flatten = valid_mask.flatten()
    invalid_mask_flatten = invalid_mask.flatten()
    delta_pair_flatten = delta_pair.reshape(numBoxes * numKeypoints, 2)
    deltaScore_flatten = deltaScore.reshape(numBoxes * numKeypoints)
    delta_pair_flatten[valid_mask_flatten, 0] = delta0
    delta_pair_flatten[valid_mask_flatten, 1] = delta1
    deltaScore_flatten[valid_mask_flatten] = s
    deltaScore_flatten[invalid_mask_flatten] = deltaScore_flatten[invalid_mask_flatten] << 16

    delta_pair = delta_pair_flatten.reshape(delta_pair.shape)
    deltaScore = deltaScore_flatten.reshape(deltaScore.shape)

    return delta_pair, deltaScore


@op_register(OpType.HeatmapMaxKeypoint)
def HeatmapMaxKeypoint_forward(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    out0 = self.outputs[0]
    out1 = self.outputs[1]
    out2 = self.outputs[2]
    score = inp0.betensor + (torch.tensor(0) if not self.quantized else torch.tensor(self.inputs[0].zerop))
    box = inp1.betensor + (torch.tensor(0) if not self.quantized else torch.tensor(self.inputs[1].zerop))
    isSoftmax = self.get_param('isSoftmax')
    if isSoftmax:
        OPT_WARN(
            'currently only support isSoftmax = 0. use all zeros output as the third output=%s, %s' % (
                self.attrs['layer_id'], self.name),
            workflow_name='forward', op_name=str(self.type))

    numBoxes = score.shape[0]
    heatmapSize = score.shape[1]
    numKeypoints = score.shape[3]
    boxInfoLength = box.shape[1]

    out0.betensor = torch.zeros([numBoxes, numKeypoints, 1], device=score.device)
    out1.betensor = torch.zeros([numBoxes, numKeypoints, 2], device=score.device)
    out2.betensor = torch.zeros([numBoxes, numKeypoints, 1], device=score.device)

    box_score = torch.permute(score, [0, 3, 1, 2]).reshape([numBoxes, numKeypoints, heatmapSize * heatmapSize]).int()
    heatmapsize_max_score, heatmapsize_max_score_index = torch.max(box_score, dim=-1, keepdims=True)
    maxIndexWidth = heatmapsize_max_score_index % heatmapSize
    maxIndexHeight = heatmapsize_max_score_index // heatmapSize
    localGrid = getLocalGrid(box_score, maxIndexWidth, maxIndexHeight, numBoxes, heatmapSize, numKeypoints)

    wRoiStart = box[:, 0]
    hRoiStart = box[:, 1]
    wRoiEnd = box[:, 2]
    hRoiEnd = box[:, 3]
    roiWidth = wRoiEnd - wRoiStart
    roiHeight = hRoiEnd - hRoiStart

    if self.quantized:
        score_scale, score_shift = self.get_param('score_scale_value'), self.get_param('score_shift_value')
        box_scale, box_shift = self.get_param('box_scale_value'), self.get_param('box_shift_value')
        delta_pair, deltaScore = solveForDelta_quant(localGrid, numBoxes, numKeypoints, heatmapsize_max_score)
        wRelativePos = maxIndexWidth * 256 + 128 + delta_pair[..., :1]
        hRelativePos = maxIndexHeight * 256 + 128 + delta_pair[..., 1:]

        wKeypointBase = wRoiStart.reshape(numBoxes, 1, 1) * heatmapSize * 256 + \
            wRelativePos * roiWidth.reshape(numBoxes, 1, 1).long()
        hKeypointBase = hRoiStart.reshape(numBoxes, 1, 1) * heatmapSize * 256 + \
            hRelativePos * roiHeight.reshape(numBoxes, 1, 1).long()

        deltaScore = linear_requantize(deltaScore, score_scale, score_shift+16, out0.zerop, out0.qmin, out0.qmax).int()
        wKeypointBase = linear_requantize(wKeypointBase, box_scale, box_shift + 8,
                                          out1.zerop, out1.qmin, out1.qmax).int()
        hKeypointBase = linear_requantize(hKeypointBase, box_scale, box_shift + 8,
                                          out1.zerop, out1.qmin, out1.qmax).int()
    else:
        delta_pair, deltaScore = solveForDelta(localGrid, numBoxes, numKeypoints, heatmapsize_max_score)
        wRelativePos = (maxIndexWidth + delta_pair[..., :1] + 0.5) / (heatmapSize)
        hRelativePos = (maxIndexHeight + delta_pair[..., 1:] + 0.5) / (heatmapSize)
        wKeypointBase = wRelativePos * roiWidth.reshape(numBoxes, 1, 1) + wRoiStart.reshape(numBoxes, 1, 1)
        hKeypointBase = hRelativePos * roiHeight.reshape(numBoxes, 1, 1) + hRoiStart.reshape(numBoxes, 1, 1)
    out0.betensor = deltaScore
    out1.betensor[..., :1] = wKeypointBase
    out1.betensor[..., 1:] = hKeypointBase

    return out0.betensor, out1.betensor, out2.betensor
