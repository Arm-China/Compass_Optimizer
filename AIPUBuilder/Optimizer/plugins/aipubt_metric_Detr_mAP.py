
# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
import numpy as np
from collections import defaultdict
from AIPUBuilder.Optimizer.plugins.aipubt_metric_mAP import BasemAPMetric


@register_plugin(PluginType.Metric, '1.0')
class DetrmAPMetric(BasemAPMetric):
    """
    This mAPMetric is used for the metric of detr model in Optimizer.
    """

    def __init__(self, class_num=90, start_label=0):
        super().__init__()
        self.class_num = int(class_num)
        self.start_label = int(start_label)
        self.conf_thresh = 0.7
        coco_80_missing_labels = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
        self.coco_80_label_map = {}
        origin_coco_labels = list(range(91))
        for l in origin_coco_labels:

            l_offset = np.sum(l > np.array(coco_80_missing_labels, np.int64))
            self.coco_80_label_map.update({l: l - l_offset})

    def __call__(self, pred, target):
        assert len(pred) == 2, OPT_FATAL(
            'please check the outputs number(should be 2)')
        batch = pred[0].shape[0]

        pred_list = []
        targets_list = []
        for pd in pred:
            pred_list.append(pd)
        for b in range(batch):
            targets = {}
            for k, v in target.items():
                targets.update({k: v[b].numpy()})
            targets_list.append(targets)

        for i in range(batch):
            predict = defaultdict()
            score = pred_list[0][i]
            probas = score.softmax(-1)[..., :-1]
            keep = probas.max(-1).values > self.conf_thresh
            scores = probas[keep]
            boxes_filter = pred_list[1][i][keep]

            label_id_list = []
            boxes_list = []
            score_list = []

            for p, (x_c, y_c, w, h) in zip(scores, boxes_filter.tolist()):
                cl = p.argmax().item()
                label_id_list.append(self.coco_80_label_map[cl]-1)
                b = [(y_c - 0.5 * h), x_c - 0.5 * w, (y_c + 0.5 * h), (x_c + 0.5 * w)]
                boxes_list.append(b)
                score_list.append(p[cl].item())
            predict.update({'label_index': label_id_list})
            predict.update({'bbox': boxes_list})
            predict.update({'confidence': score_list})

            predict.update({'image_name': targets_list[i]['image_name']})
            BasemAPMetric.extract_obj_all_class(predict, self.predicts)
            BasemAPMetric.extract_obj_all_class(targets_list[i], self.targets)

    def reset(self):
        super().reset()

    def compute(self):
        self.eval_mAP(self.predicts, self.targets, self.class_num, 0)
        return self.mAP

    def report(self):
        return "mAP accuracy is %f" % (self.compute())
