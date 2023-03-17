# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.plugins.aipubt_metric_mAP import mAPMetric


@register_plugin(PluginType.Metric, '1.0')
class FasterRcnnCaffemAPMetric(mAPMetric):
    """
    This FasterRcnnCaffemAPMetric is used for the metric of fasterrcnn_caffe model in Optimizer.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def __call__(self, pred, target):
        pred_post = [pred[4], pred[3], pred[5], pred[6], pred[7], pred[8]]
        super().__call__(pred_post, target)

    def reset(self):
        super().reset()

    def compute(self):
        result = super().compute()
        return result

    def report(self):
        return "fasterrcnn caffe mAP accuracy is %f" % (self.compute())


@register_plugin(PluginType.Metric, '1.0')
class FasterRcnnTFmAPMetric(mAPMetric):
    """
    This FasterRcnnTFmAPMetric is used for the metric of fasterrcnn_tensorflow model in Optimizer.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, pred, target):
        pred_post = [pred[6], pred[5], pred[7], pred[8], pred[9], pred[10]]
        super().__call__(pred_post, target)

    def reset(self):
        super().reset()

    def compute(self):
        result = super().compute()
        return result

    def report(self):
        return "fasterrcnn tensorflow mAP accuracy is %f" % (self.compute())
