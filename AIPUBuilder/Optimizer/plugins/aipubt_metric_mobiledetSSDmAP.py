# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from AIPUBuilder.Optimizer.plugins.aipubt_metric_mAP import mAPMetric, BasemAPMetric
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.plugins.aipubt_op_ssd_postprocess import SSDPostProcess
from collections import defaultdict
import numpy as np
import os


@register_plugin(PluginType.Metric, '1.0')
class MobileDetSSDmAPMetric(mAPMetric):
    """
    This MobileDetSSDmAPMetric is used for the metric of mobiledet_ssd_tflite models in Optimizer.
    This plugin firstly do postprocess which including decodebox and nms and then computes the mAP of SSD models.
    We assume the iou_threshold=0.5.
    """

    iou_thresh = 0.5

    def __init__(self, anchor_path=None, model_name=None):
        super().__init__()
        self.coco_80_label_map = self.get_label_map()

        if anchor_path is not None:
            if not os.path.exists(anchor_path):
                OPT_ERROR(f"please check the anchor_path={anchor_path} which is not existed")
            anchor = np.load(anchor_path)
            self.db_params = {'anchor': anchor}
        # self.postprocess = SSDPostProcess(decodebox_params=db_params)
        self.model_name = model_name

    def __call__(self, backbone_pred, target):
        assert len(backbone_pred) == 2, OPT_FATAL('please check the outputs number(should be 2)')

        out_db, out_nms = SSDPostProcess(decodebox_params=self.db_params)(
            input_tensors=[backbone_pred[1], backbone_pred[0]])

        pred = out_db + out_nms
        batch = pred[2].shape[0]

        pred_list = []
        targets_list = []
        for pd in pred:
            pred_list.append(pd.cpu().numpy())
        for b in range(batch):
            targets = {}
            for k, v in target.items():
                targets.update({k: v[b].cpu().numpy()})
            targets_list.append(targets)

        for i in range(batch):
            predict = defaultdict()
            obj_class_num = pred_list[2][i]
            label_pre_class = pred_list[4][i]
            boxes = pred_list[5][i]
            box_num_pre_class = pred_list[6][i]
            scores = pred_list[7][i]
            keep = pred_list[8][i]

            label_id_list = []
            boxes_list = []
            score_list = []
            all_box_idx = 0
            for cid in range(int(obj_class_num[0])):
                box_num_cur_class = int(box_num_pre_class[cid])
                label_id = label_pre_class[cid]

                label_id_list.extend([label_id]*box_num_cur_class)
                boxes_list.extend([boxes[all_box_idx+box_idx]
                                   for box_idx in range(box_num_cur_class)])
                score_list.extend([scores[all_box_idx+box_idx]
                                   for box_idx in range(box_num_cur_class)])

                all_box_idx = all_box_idx + box_num_cur_class
            label_id_list = [self.coco_80_label_map[ll+1] - 1 for ll in label_id_list]
            predict.update({'label_index': label_id_list})
            predict.update({'bbox': boxes_list})
            predict.update({'confidence': score_list})
            predict.update({'image_name': targets_list[i]['image_name']})
            BasemAPMetric.extract_obj_all_class(predict, self.predicts)
            BasemAPMetric.extract_obj_all_class(targets_list[i], self.targets)

    def reset(self):
        super().reset()

    def compute(self):
        self.eval_mAP(self.predicts, self.targets, 80, 0)
        return self.mAP

    def report(self):
        return "SSDVOC mAP accuracy is %f" % (self.compute())

    def get_label_map(self):
        origin_coco_labels = list(range(91))
        coco_80_missing_labels = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
        coco_80_label_map = {}
        for l in origin_coco_labels:
            l_offset = np.sum(l > np.array(coco_80_missing_labels, np.int64))
            coco_80_label_map.update({l: l - l_offset})
        return coco_80_label_map
