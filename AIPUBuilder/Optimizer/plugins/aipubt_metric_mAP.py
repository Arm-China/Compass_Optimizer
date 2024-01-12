# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import numpy as np
from collections import defaultdict


class BasemAPMetric(OptBaseMetric):
    iou_thresh = 0.5

    def __init__(self):
        super().__init__()
        self.preds_bboxes = []
        self.preds_labels = []
        self.preds_scores = []
        self.gt_bboxes = []
        self.gt_labels = []
        self.AP = 0
        self.mAP = 0
        self.predicts = defaultdict()
        self.targets = defaultdict()

    def reset(self):
        self.AP = 0
        self.mAP = 0
        self.predicts = defaultdict()
        self.targets = defaultdict()

    @staticmethod
    def extract_obj_all_class(obj, collector):
        if 'image_name' not in obj or 'label_index' not in obj:
            return

        image_name = int(obj['image_name'])
        label_index = obj['label_index']
        bbox = obj['bbox']
        if isinstance(label_index, np.ndarray):
            label_index = label_index.reshape([-1]).tolist()
        if isinstance(bbox, np.ndarray):
            bbox = bbox.reshape([-1, 4]).tolist()

        for idx in range(len(label_index)):
            label_idx = int(label_index[idx])
            bbox_exist = False

            if label_idx not in list(collector.keys()):
                collector.update({label_idx: {image_name: {'bbox': [bbox[idx]]}}})
            else:
                if image_name not in list(collector[label_idx].keys()):
                    collector[label_idx].update({image_name: {'bbox': [bbox[idx]]}})
                else:
                    for bbox_i in collector[label_idx][image_name]['bbox']:
                        if isinstance(bbox_i, np.ndarray):
                            if (bbox_i == bbox[idx]).all():
                                bbox_exist = True
                                break
                        elif bbox_i == bbox[idx]:
                            bbox_exist = True
                            break
                    if not bbox_exist:
                        collector[label_idx][image_name]['bbox'].append(bbox[idx])

            if 'confidence' in obj and not bbox_exist:
                confidence = obj['confidence']
                if 'confidence' not in list(collector[label_idx][image_name].keys()):
                    collector[label_idx][image_name].update({'confidence': [confidence[idx]]})
                else:
                    collector[label_idx][image_name]['confidence'].append(confidence[idx])

    @staticmethod
    def cal_rec_prec(tp, fp, num_bbgt):
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        if (tp + fp).min() == 0.0:
            OPT_WARN('TP add FP is zero')
        recall = tp / num_bbgt
        precision = tp / (tp + fp)
        return recall, precision

    @staticmethod
    def cal_iou_yxyx(bbox_det, bbox_gt):
        '''
        :param bbox_det: n * 4, [[ymin, xmin, ymax, xmax],  [ymin, xmin, ymax, xmax], ...]
        :param bbox_gt:
        :return:
        '''
        ymin = np.maximum(bbox_det[..., 0], bbox_gt[..., 0])
        xmin = np.maximum(bbox_det[..., 1], bbox_gt[..., 1])
        ymax = np.minimum(bbox_det[..., 2], bbox_gt[..., 2])
        xmax = np.minimum(bbox_det[..., 3], bbox_gt[..., 3])

        width = np.maximum(xmax - xmin, 0)
        height = np.maximum(ymax - ymin, 0)
        inter = height * width

        union = (bbox_det[..., 2] - bbox_det[..., 0]) * (bbox_det[..., 3] - bbox_det[..., 1]) + \
                (bbox_gt[..., 2] - bbox_gt[..., 0]) * (bbox_gt[..., 3] - bbox_gt[..., 1]) - inter

        overlap = inter / union
        return overlap

    @staticmethod
    def cal_iou_xywh(box1, box2):
        tb = min(box1[..., 0]+0.5*box1[..., 2], box2[..., 0]+0.5*box2[..., 2]) - \
            max(box1[..., 0]-0.5*box1[..., 2], box2[..., 0]-0.5*box2[..., 2])
        lr = min(box1[..., 1]+0.5*box1[..., 3], box2[..., 1]+0.5*box2[..., 3]) - \
            max(box1[..., 1]-0.5*box1[..., 3], box2[..., 1]-0.5*box2[..., 3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb*lr
        return intersection / (box1[..., 2]*box1[..., 3] + box2[..., 2]*box2[..., 3] - intersection)

    @staticmethod
    def cal_AP(recall, precision):
        rec = np.concatenate(([0.], recall, [1.0]))
        prec = np.concatenate(([0.], precision, [0.]))

        for prec_idx in range(prec.shape[0]-1, 0, -1):
            prec[prec_idx-1] = np.maximum(prec[prec_idx], prec[prec_idx-1])

        rec_idx = np.where(rec[1:] != rec[:-1])[0]
        ap = np.sum((rec[rec_idx + 1] - rec[rec_idx]) * prec[rec_idx + 1])

        return ap

    @staticmethod
    def cal_mAP(ap):
        mAP = np.nanmean(ap)
        return mAP

    def eval_mAP(self, prediction, gt, class_num, start_label_id=0):
        '''
        :param pred: {labeld_index: {label_name:
                                     image_name: {bbox: , confidence: }}}
        :param gt:
        :param class_num:
        :param start_label_id:
        :return:
        '''

        def extract_obj_per_class(target, label_index):
            return target[label_index] if label_index in target else None

        ap = []
        for class_idx in range(class_num):
            if class_idx < start_label_id:
                continue
            prediction_per_class = extract_obj_per_class(prediction, class_idx)
            gt_per_class = extract_obj_per_class(gt, class_idx)

            if not (prediction_per_class is None or gt_per_class is None):
                image_names = list(prediction_per_class.keys())

                num_bbgt = 0
                for n in gt_per_class.keys():
                    nbbgt = len(gt_per_class[n]['bbox'])
                    gt_per_class[n].update({'match': [False] * nbbgt})
                    num_bbgt += nbbgt

                # get the detection result
                # confidence: [img_num, MAX_NMS_BOX_NUM]
                # bboxes: [img_num, MAX_NMS_BOX_NUM, 4]
                # gtbboxes: [img_num, MAX_NMS_BOX_NUM, 4]
                imagename_list = []
                confidences = []
                bboxes = []
                for image_name in image_names:
                    c = prediction_per_class[image_name]['confidence']
                    bb = prediction_per_class[image_name]['bbox']
                    for i in range(len(c)):
                        imagename_list.append(image_name)
                        confidences.append(c[i])
                        bboxes.append(bb[i])
                confidences = np.array(confidences)
                bboxes = np.array(bboxes)

                # sort all the detection boxes confidences for current class
                sorted_inds = np.argsort(-confidences)
                bboxes = bboxes[sorted_inds]
                imagename_list = [imagename_list[idx] for idx in sorted_inds]

                # the count of all the detection boxes for current class
                num_pbb = len(confidences)
                TP = np.zeros(num_pbb)
                FP = np.zeros(num_pbb)
                for idx in range(num_pbb):
                    bb = bboxes[idx]
                    name = imagename_list[idx]
                    if name not in gt_per_class:
                        FP[idx] = 1
                    else:
                        gt_class_image = gt_per_class[name]
                        bbox_gt = np.array(gt_class_image['bbox'])
                        iou = BasemAPMetric.cal_iou_yxyx(bb, bbox_gt)
                        midx = np.argmax(iou)
                        if iou[midx] > BasemAPMetric.iou_thresh:
                            if not gt_class_image['match'][midx]:
                                gt_class_image['match'][midx] = True
                                TP[idx] = 1
                            else:
                                FP[idx] = 1
                        else:
                            FP[idx] = 1
                recall, precision = BasemAPMetric.cal_rec_prec(
                    TP, FP, num_bbgt)
                ap.append(BasemAPMetric.cal_AP(recall, precision))
            elif prediction_per_class is None and gt_per_class is None:
                ap.append(np.nan)
            elif gt_per_class is not None:
                ap.append(0)
        self.mAP = BasemAPMetric.cal_mAP(ap)


@register_plugin(PluginType.Metric, '1.0')
class mAPMetric(BasemAPMetric):
    """
    This mAPMetric is used for the metric of vgg_fasterrcnn model in Optimizer.
    """

    def __init__(self, class_num=90, start_label=0):
        super().__init__()
        self.class_num = int(class_num)
        self.start_label = int(start_label)

    def __call__(self, pred, target, *args):
        assert len(pred) == 6, OPT_FATAL("please check the outputs number(should be 6 in mAPMetric)")

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
            obj_class_num = pred_list[0][i]
            label_pre_class = pred_list[1][i]
            boxes = pred_list[2][i]
            box_num_pre_class = pred_list[3][i]
            scores = pred_list[4][i]
            keep = pred_list[5][i]

            label_id_list = []
            boxes_list = []
            score_list = []
            all_box_idx = 0
            for cid in range(int(obj_class_num[0])):
                box_num_cur_class = int(box_num_pre_class[cid])
                label_id = int(label_pre_class[cid])

                if label_id < self.start_label:
                    all_box_idx += box_num_cur_class
                    continue

                label_id_list.extend([label_id]*box_num_cur_class)
                boxes_list.extend([boxes[all_box_idx+box_idx]
                                   for box_idx in range(box_num_cur_class)])
                score_list.extend([scores[all_box_idx+box_idx]
                                   for box_idx in range(box_num_cur_class)])

                all_box_idx = all_box_idx + box_num_cur_class

            label_id_list = [l - self.start_label for l in label_id_list]
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
        return f"mAP accuracy is {self.compute()}"
