# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from AIPUBuilder.Optimizer.plugins.aipubt_metric_mAP import mAPMetric
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import OPT_FATAL


@register_plugin(PluginType.Metric, '1.0')
class SSDmAPMetric(mAPMetric):
    """
    This SSDmAPMetric is used for the metric of SSD models in Optimizer.
    This plugin computes the mAP of SSD models.
    We assume the iou_threshold=0.5.
    """

    def __init__(self, class_num=90, start_id=0):
        super().__init__(class_num, start_id)

    def __call__(self, pred, target):
        assert len(pred) == 9, OPT_FATAL('please check the outputs number(should be 9)')
        pred_post = [pred[2], pred[4], pred[5], pred[6], pred[7], pred[8]]
        super().__call__(pred_post, target)

    def reset(self):
        super().reset()

    def compute(self):
        self.mAP = super().compute()
        return self.mAP

    def report(self):
        return "SSD mAP accuracy is %f" % (self.compute())
