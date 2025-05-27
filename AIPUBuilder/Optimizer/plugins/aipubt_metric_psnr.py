# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

import torch
import cv2
import numpy as np


@register_plugin(PluginType.Metric, '1.0')
# this psnr can check if image can be recovered or not.
# if psnr is too low, it means image can't be recovered
# but cosine similarity maybe high, so only cosine similarity can't reflect net' performance
# so we design a simple rule:
# cos similarity need multipled by factor
# less psnr, less factor
class PsnrMetric(OptBaseMetric):
    """
    This PsnrMetric is used for the metric of espcn_tensorflow model in Optimizer.

    this PsnrMetric can check if image can be recovered or not.
    if psnr is too low, it means image can't be recovered, but cosine similarity maybe high,
    so only cosine similarity can't reflect net' performance.
    for reason given above, we design a simple rule: cos similarity need multipled by factor; less psnr, less factor
    """

    def __init__(self):
        self.loss = torch.nn.MSELoss(reduction='sum')
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.sim = []
        self.sim_psnr = []

        self.showimage = False  # for debug
        self.dumpimage_dir = "./"

    def __call__(self, pred, target):
        pred = pred[0]
        # for sr model, h,w size must be much greater than 3,or it's data layout is nchw
        if pred.shape[1] < 4:
            pred = torch.clamp(pred.permute(0, 2, 3, 1)*255, 0, 255)
        shape = pred.shape
        x, y = [], []
        if target.shape[-1] > 1:
            pred_y = torch.flatten(pred)
            target_y = torch.flatten(target[:, :, :, 0])
            target_u = torch.flatten(target[:, :, :, 2])
            target_v = torch.flatten(target[:, :, :, 1])
            target_r = target_y + 1.4075 * (target_v - 128)
            y = torch.clamp(target_r, 0, 255)
            if self.showimage == True:

                g = target_y - 0.3455 * (target_u - 128) - 0.7169 * (target_v - 128)
                g = torch.clamp(g, 0, 255)

                b = target_y + 1.779 * (target_u - 128)
                b = torch.clamp(b, 0, 255)
                img_ref = torch.cat((y.reshape(shape[1], shape[2], 1), g.reshape(
                    shape[1], shape[2], 1), b.reshape(shape[1], shape[2], 1)), dim=2)

            x = pred_y + 1.4075 * (target_v - 128)
            x = torch.clamp(x, 0, 255)
            if self.showimage == True:
                g = pred_y - 0.3455 * (target_u - 128) - 0.7169 * (target_v - 128)
                g = torch.clamp(g, 0, 255)

                b = pred_y + 1.779 * (target_u - 128)
                b = torch.clamp(b, 0, 255)
                img_pred = torch.cat((x.reshape(shape[1], shape[2], 1), g.reshape(
                    shape[1], shape[2], 1), b.reshape(shape[1], shape[2], 1)), dim=2)

                image_ref = img_ref.cpu().numpy().astype(np.uint8)
                img_pred = img_pred.cpu().numpy().astype(np.uint8)
                cv2.imwrite(self.dumpimage_dir+"image_ref.png", image_ref)
                cv2.imwrite(self.dumpimage_dir+"image_pred.png", img_pred)
        else:
            x = torch.clamp(pred, 0, 255).flatten()
            y = torch.clamp(target, 0, 255).flatten()
        mse = self.loss(x, y)/shape[1]/shape[2]
        psnr = 10*torch.log10((255.0*255/mse))
        self.sim_psnr.append(psnr)
        self.sim.append(self.cos(x, y))

    def reset(self):
        self.sim = []
        self.sim_psnr = []

    def compute(self):
        # shape of tensor self.sim is rank 1 as x, y above have iterate all dims and reshape
        t = torch.Tensor(self.sim_psnr)
        psnr = float(torch.mean(t, 0))
        factor = 1.0
        if psnr < 10:
            factor = 0.1
        elif psnr < 20:
            factor = 0.85
        else:
            factor = 1.0
        t = torch.Tensor(self.sim)*factor
        return float(torch.mean(t, 0))

    def report(self):
        return "cosine similarity is %f" % (self.compute())
