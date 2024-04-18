
# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
import numpy as np
from collections import defaultdict
from AIPUBuilder.Optimizer.plugins.aipubt_metric_mAP import BasemAPMetric


@register_plugin(PluginType.Metric, '1.0')
class FcosmAPMetric(BasemAPMetric):
    """
    This mAPMetric is used for the metric of fcos model in Optimizer.
    """

    def __init__(self, class_num=80, start_label=0):
        super().__init__()
        self.class_num = int(class_num)
        self.start_label = int(start_label)
        self.conf_thresh = 0.2
        self.shape = [800, 1216]
        self.iou_threshold = 0.5

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
            score = pred_list[1][i].transpose(1, 0)
            # probas = score.softmax(-1)[..., :-1]
            keep = score.max(-1).values > self.conf_thresh
            scores = score[keep]
            if not scores.size:
                continue
            boxes_filter = pred_list[0][i][keep]

            label_id_list = []
            boxes_list = []
            score_list = []

            for p, (y0, x0, y1, x1) in zip(scores, boxes_filter.tolist()):
                cl = p.argmax().item()
                label_id_list.append(cl)
                b = [x0 / self.shape[0], y0 / self.shape[1], x1 / self.shape[0], y1 / self.shape[1]]
                boxes_list.append(b)
                score_list.append(p[cl].item())
            try:
                pred_bbox = np.concatenate([boxes_list, np.array(score_list)[..., np.newaxis],
                                           np.array(label_id_list)[..., np.newaxis]], axis=-1)
                pred_bbox = np.array(self.nms(pred_bbox))
                predict.update({'label_index': pred_bbox[:, 5]})
                predict.update({'bbox': pred_bbox[:, :4]})
                predict.update({'confidence': pred_bbox[:, 4]})
            except:
                predict.update({'label_index': []})
                predict.update({'bbox': []})
                predict.update({'confidence': []})

            predict.update({'image_name': targets_list[i]['image_name']})
            BasemAPMetric.extract_obj_all_class(predict, self.predicts)
            BasemAPMetric.extract_obj_all_class(targets_list[i], self.targets)

    def nms(self, bboxes):
        """
        param bboxes: (xmin, ymin, xmax, ymax, score, class)
        """
        classes_in_img = list(set(bboxes[..., 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[..., 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[..., 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate(
                    [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = BasemAPMetric.cal_iou_yxyx(
                    best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)
                iou_mask = iou > self.iou_threshold
                weight[iou_mask] = 0.0

                cls_bboxes[:, 4] = cls_bboxes[..., 4] * weight
                score_mask = cls_bboxes[..., 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes

    def reset(self):
        super().reset()

    def compute(self):
        self.eval_mAP(self.predicts, self.targets, self.class_num, 0)
        return self.mAP

    def report(self):
        return "mAP accuracy is %f" % (self.compute())
