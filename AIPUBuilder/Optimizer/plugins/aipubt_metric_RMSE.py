# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import os
import sys
import cv2
import torch
import numpy as np

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *


@register_plugin(PluginType.Metric, '0.01')
class RMSEMetric(OptBaseMetric):
    """
    This RMSEMetric is used for the metric of dinov2-small-nyu model in Optimizer.
    """

    def __init__(self):
        self.total = 0
        self.total_loss = 0.0

    def __call__(self, pred, target):
        image_size = target.shape[1:]
        prediction = torch.nn.functional.interpolate(
            pred[0].unsqueeze(1),
            size=image_size,
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

        target = target.cpu().numpy().reshape(image_size)
        loss = np.sqrt(np.mean((prediction - target / 1000.) ** 2))
        # print(f"{self.total}: RMSE loss is {loss}")
        self.total += 1
        self.total_loss += loss

    def reset(self):
        self.total = 0
        self.total_loss = 0.0

    def compute(self):
        average_loss = self.total_loss / self.total
        return average_loss

    def report(self):
        return "rmse accuracy is %f" % (self.compute())
