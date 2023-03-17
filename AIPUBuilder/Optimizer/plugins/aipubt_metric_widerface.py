
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch
import numpy as np


@register_plugin(PluginType.Metric, '1.0')
class WiderFaceMetric(OptBaseMetric):
    """
    This WiderFaceMetric is used for the metric of widerface dataset in Optimizer.
    using for lightface and centerface,retinaface

    """

    def __init__(self):

        self.image_size = [640, 640]
        self.scale = torch.Tensor([1, 1, 1, 1])

        self.nms_threshold = 0.3
        self.box_pred = []
        self.box_gt = []
        self.iou_thresh = 0.5
        self.heatmap_thresh = 0.123

    def __call__(self, pred, target):
        batch_size = pred[0].shape[0]

        for batch in range(batch_size):
            heatmap, scale, offset, lms = pred[0][batch:batch+1].cpu().numpy(), pred[1][batch:batch+1].cpu().numpy(),\
                pred[2][batch:batch+1].cpu().numpy(), pred[3][batch:batch +
                                                              1].cpu().numpy()
            dets, lms = self.postprocess(
                heatmap, lms, offset, scale, self.heatmap_thresh)

            boxes = []
            for box in target['bbox'][batch].cpu().numpy():
                boxes.append(box)
            self.box_pred.append(dets)
            self.box_gt.append([boxes,
                                target['easy'][batch].cpu().numpy(),
                                target['medium'][batch].cpu().numpy(),
                                target['hard'][batch].cpu().numpy()])

    def postprocess(self, heatmap, lms, offset, scale, threshold):
        dets, lms = self.decode(heatmap, scale, offset, lms,
                                (self.image_size[0], self.image_size[1]), threshold=threshold)
        if len(dets) == 0:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            lms = np.empty(shape=[0, 10], dtype=np.float32)

        return dets, lms

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        boxes, lms = [], []

        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * \
                    4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 /
                             2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]),
                              min(y1 + s0, size[0]), s])
                lm = []
                for j in range(5):
                    lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                    lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = self.nms(boxes)
            boxes = boxes[keep, :]
            lms = np.asarray(lms, dtype=np.float32)
            lms = lms[keep, :]

        return boxes, lms

    # ##################################################################
    # Inputs
    #     box_scores (N, 5): boxes in (x0,y0,x1,y1) and scores.
    #     top_k: keep top_k results. If k <= 0, keep all the results.
    #     max_box_number: only consider the candidates with the highest scores.
    # outputs:
    #      a list of keep_idx of the kept boxes
    # #########################################################################

    def nms(self, box_scores, top_k=-1, max_box_number=200):

        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        max_box_number = boxes.shape[0]
        keep = []
        order_idx = np.argsort(scores)
        order_idx = order_idx[-max_box_number:]
        while len(order_idx) > 0:
            current = order_idx[-1]
            keep.append(current)
            if 0 < top_k == len(keep) or len(order_idx) == 1:
                break
            current_box = boxes[current, :]
            order_idx = order_idx[:-1]
            other_boxes = boxes[order_idx, :]
            ref_box = np.expand_dims(current_box, axis=0)

            x0_y0 = np.maximum(other_boxes[..., :2], ref_box[..., :2])
            x1_y1 = np.minimum(other_boxes[..., 2:], ref_box[..., 2:])
            hw = np.clip(x1_y1 - x0_y0, 0.0, None)
            inter_area = hw[..., 0] * hw[..., 1]
            hw = np.clip(other_boxes[..., 2:] - other_boxes[..., :2], 0.0, None)
            area0 = hw[..., 0] * hw[..., 1]
            hw = np.clip(ref_box[..., 2:] - ref_box[..., :2], 0.0, None)
            area1 = hw[..., 0] * hw[..., 1]
            iou = inter_area / (area0 + area1 - inter_area + 1e-7)

            order_idx = order_idx[iou <= self.nms_threshold]

        return keep

    def reset(self):
        pass

    def compute(self):
        return self.eval_map(self.box_pred, self.box_gt)

    def eval_map(self, pred, target):
        aps = []
        self.normalize_score(pred)

        thresh_num = 500
        # event type has 3(easy,medium, hard)
        for event_id in range(3):

            count_face = 0
            precsion_recall_curve = np.zeros((thresh_num, 2)).astype('float')
            for i in range(len(target)):

                pred_info = pred[i]
                gt_bbx_list = target[i]
                gt_boxes = gt_bbx_list[0][0]
                keep_index = gt_bbx_list[1+event_id][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1
                pred_recall, proposal_list = self.image_eval(
                    pred_info, gt_boxes, ignore, self.iou_thresh)

                presicion_recall_info = self.once_pred_presicion_recall_info(
                    thresh_num, pred_info, proposal_list, pred_recall)

                precsion_recall_curve += presicion_recall_info
            precsion_recall_curve = self.dataset_presicion_recall_info(thresh_num, precsion_recall_curve, count_face)

            precision = precsion_recall_curve[:, 0]
            recall = precsion_recall_curve[:, 1]

            ap = self.AP(recall, precision)
            aps.append(ap)
        return aps

    def AP(self, recall, precision):

        rec = np.concatenate(([0.], recall, [1.0]))
        prec = np.concatenate(([0.], precision, [0.]))

        for prec_idx in range(prec.shape[0]-1, 0, -1):
            prec[prec_idx-1] = np.maximum(prec[prec_idx], prec[prec_idx-1])

        rec_idx = np.where(rec[1:] != rec[:-1])[0]
        ap = np.sum((rec[rec_idx + 1] - rec[rec_idx]) * prec[rec_idx + 1])
        return ap

    def once_pred_presicion_recall_info(self, thresh_num, pred_info, proposal_list, pred_recall):
        pr_info = np.zeros((thresh_num, 2)).astype('float')
        for t in range(thresh_num):

            thresh = 1 - (t+1)/thresh_num
            r_index = np.where(pred_info[:, 4] >= thresh)[0]
            if len(r_index) == 0:
                pr_info[t, 0] = 0
                pr_info[t, 1] = 0
            else:
                r_index = r_index[-1]
                p_index = np.where(proposal_list[:r_index+1] == 1)[0]
                pr_info[t, 0] = len(p_index)
                pr_info[t, 1] = pred_recall[r_index]
        return pr_info

    def dataset_presicion_recall_info(self, thresh_num, pr_curve, count_face):
        _pr_curve = np.zeros((thresh_num, 2))
        for i in range(thresh_num):
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
            _pr_curve[i, 1] = pr_curve[i, 1] / count_face
        return _pr_curve
    ####################################################
    # norm score according to all pred's score
    # pred is list as {index: [[x0,y0,x1,y1,score]]}
    #####################################################

    def normalize_score(self, pred):
        max_score = 0
        min_score = 1
        for i in range(len(pred)):
            if len(pred[i]) == 0:
                continue
            _min = np.min(pred[i][:, -1])
            _max = np.max(pred[i][:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

        dist = max_score - min_score
        for i in range(len(pred)):
            if len(pred[i]) == 0:
                continue
            pred[i][:, -1] = (pred[i][:, -1] - min_score)/dist
    # #########################################################
    # calcuate iou between pred boxes and target boxes
    # ----------
    # boxes: (N, 4) (x0.y0.x1.y1)
    # target_boxes: (K, 4) (x,y,w,h)
    #
    #
    # ious: (N, K) matrix of iou between boxes and target_boxes
    # ###############################################################

    def get_boxes_iou(self,
                      boxes,
                      target_boxes):

        N = boxes.shape[0]
        K = target_boxes.shape[0]
        ious = np.zeros((N, K), dtype=np.float)

        for k in range(K):
            box_area = (target_boxes[k, 2] + 1) * \
                       (target_boxes[k, 3] + 1)
            for n in range(N):
                iw = min(boxes[n, 2], target_boxes[k, 0] + target_boxes[k, 2]) - \
                    max(boxes[n, 0], target_boxes[k, 0]) + 1
                if iw > 0:
                    ih = min(boxes[n, 3], target_boxes[k, 1] + target_boxes[k, 3]) - \
                        max(boxes[n, 1], target_boxes[k, 1]) + 1

                    if ih > 0:
                        union_area = (boxes[n, 2] - boxes[n, 0] + 1) * \
                            (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih

                        ious[n, k] = iw * ih / union_area
        return ious
    ###############################################################
    # single image evaluation
    # pred: Nx5 (x0,y0,x1,y1,score)
    # gt: Kx4 (x,y,w,h)
    #
    # ##############################################################

    def image_eval(self, pred, gt, ignore, iou_thresh):

        pred_recall = np.zeros(pred.shape[0])
        recall_list = np.zeros(gt.shape[0])
        proposal_list = np.ones_like(pred_recall)

        ious = self.get_boxes_iou(pred[:, :4], gt)

        for h in range(pred.shape[0]):

            iou = ious[h, ...]
            max_iou, max_idx = iou.max(), iou.argmax()
            if max_iou >= iou_thresh:
                if ignore[max_idx] == 0:
                    recall_list[max_idx] = -1
                    proposal_list[h] = -1
                elif recall_list[max_idx] == 0:
                    recall_list[max_idx] = 1

            r_keep_index = np.where(recall_list == 1)[0]
            pred_recall[h] = len(r_keep_index)
        return pred_recall, proposal_list

    def report(self):
        aps = self.compute()
        return ("mAP accuracy easy is %f medium is %f hard is %f" % (aps[0], aps[1], aps[2]))
