# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch
from math import ceil
from itertools import product as product
import numpy as np


@register_plugin(PluginType.Metric, '1.0')
class faceboxMetric(OptBaseMetric):
    """
    This faceboxMetric is used for the metric of FaceBoxes_onnx model in Optimizer.
    The input image size of facebox model is 1024x1024.
    """

    def __init__(self):
        self.cfg = {
            'name': 'FaceBoxes',
            # 'min_dim': 1024,
            # 'feature_maps': [[32, 32], [16, 16], [8, 8]],
            # 'aspect_ratios': [[1], [1], [1]],
            'min_sizes': [[32, 64, 128], [256], [512]],
            'steps': [32, 64, 128],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': True
        }
        self.image_size = [1024, 1024]
        self.scale = torch.Tensor(
            [self.image_size[1], self.image_size[0], self.image_size[1], self.image_size[0]])
        self.confidence_threshold = 0.05
        self.nms_threshold = 0.3
        self.box_pred = []
        self.box_gt = []
        self.iou_thresh = 0.5

    def __call__(self, pred, target):
        batch_size = pred[0].shape[0]
        for batch in range(batch_size):
            delta, conf = pred[0][batch:batch +
                                  1].cpu(), pred[1][batch:batch+1].cpu()
            anchor_data = self.anchors_data().data
            boxes = self.decode(delta.data.squeeze(
                0), anchor_data, self.cfg['variance'])
            boxes = boxes * self.scale
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

            # ignore low scores
            inds = np.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = list(scores.argsort()[::-1][:5000])
            boxes = boxes[order]
            scores = scores[order]
            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = self.nms(dets, self.nms_threshold)
            dets = dets[keep, :]

            # store the data and labels
            boxes = []
            for box in target[batch].cpu().numpy():
                if not all(box.flatten()):
                    break
                boxes.append(box)
            self.box_pred.append(dets)
            self.box_gt.append(boxes)

    def decode(self, delta, anchos, variances):
       # get box
        boxes = torch.cat((
            anchos[:, :2] + delta[:, :2] * variances[0] * anchos[:, 2:],
            anchos[:, 2:] * torch.exp(delta[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def anchors_data(self):
        feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)]
                        for step in self.cfg['steps']]
        steps = self.cfg['steps']
        clip = self.cfg['clip']
        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.cfg['min_sizes'][k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x*steps[k]/self.image_size[1]
                                    for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y*steps[k]/self.image_size[0]
                                    for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x*steps[k]/self.image_size[1]
                                    for x in [j+0, j+0.5]]
                        dense_cy = [y*steps[k]/self.image_size[0]
                                    for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * steps[k] / self.image_size[1]
                        cy = (i + 0.5) * steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if clip:
            output.clamp_(max=1, min=0)
        return output

    def nms(self, dets, threshold):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return keep

    def reset(self):
        pass

    def compute(self):
        return self.eval_map(self.box_pred, self.box_gt)

    def eval_map(self, pred, gt):
        ap = []

        if not (pred is None or gt is None):
            imagename_list = []
            confidences = []
            bboxes = []
            gt_dict = {}
            num_bbgt = 0
            for i in range(len(pred)):
                c = pred[i][:, -1]
                bb = pred[i][:, :-1]
                for k in range(len(c)):
                    imagename_list.append(i)
                    confidences.append(c[k])
                    bboxes.append(bb[k])

                nbbgt = len(gt[i])
                gt_dict[i] = {
                    'match': [False] * nbbgt,
                    'bbox': gt[i] * self.scale.cpu().numpy()
                }
                num_bbgt += nbbgt
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
                if name not in gt_dict:
                    FP[idx] = 1
                else:
                    gt_class_image = gt_dict[name]
                    bbox_gt = np.array(gt_class_image['bbox'])
                    iou = self.iou(bb, bbox_gt)
                    midx = np.argmax(iou)
                    if iou[midx] > self.iou_thresh:
                        if not gt_class_image['match'][midx]:
                            gt_class_image['match'][midx] = True
                            TP[idx] = 1
                        else:
                            FP[idx] = 1
                            # already have matched
                    else:
                        FP[idx] = 1
                        # IOU smaller than threshold
            recall, precision = self.cal_rec_prec(TP, FP, num_bbgt)
            ap.append(self.AP(recall, precision))
        elif pred is None and gt is None:
            ap.append(np.nan)
        elif gt is not None:
            ap.append(0)
        return ap[0]

    @staticmethod
    def iou(bbox_det, bbox_gt):

        ymin = np.maximum(bbox_det[0], bbox_gt[:, 0])
        xmin = np.maximum(bbox_det[1], bbox_gt[:, 1])
        ymax = np.minimum(bbox_det[2], bbox_gt[:, 2])
        xmax = np.minimum(bbox_det[3], bbox_gt[:, 3])

        width = np.maximum(xmax - xmin, 0)
        height = np.maximum(ymax - ymin, 0)
        inter = height * width

        union = (bbox_det[2] - bbox_det[0]) * (bbox_det[3] - bbox_det[1]) + \
                (bbox_gt[:, 2] - bbox_gt[:, 0]) * \
            (bbox_gt[:, 3] - bbox_gt[:, 1]) - inter

        overlap = inter / union

        return overlap

    @staticmethod
    def cal_rec_prec(tp, fp, num_bbgt):

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        if (tp+fp).min() == 0.0:
            OPT_WARN('TP add FP is zero')
        recall = tp / num_bbgt
        precision = tp / (tp + fp)
        return recall, precision

    @staticmethod
    def AP(recall, precision):

        rec = np.concatenate(([0.], recall, [1.0]))
        prec = np.concatenate(([0.], precision, [0.]))

        for prec_idx in range(prec.shape[0]-1, 0, -1):
            prec[prec_idx-1] = np.maximum(prec[prec_idx], prec[prec_idx-1])

        rec_idx = np.where(rec[1:] != rec[:-1])[0]
        ap = np.sum((rec[rec_idx + 1] - rec[rec_idx]) * prec[rec_idx + 1])

        return ap

    def report(self):
        return "mAP accuracy is %f" % (self.compute())
