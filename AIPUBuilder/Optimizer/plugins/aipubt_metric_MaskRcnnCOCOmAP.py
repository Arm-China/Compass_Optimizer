# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import torch
import numpy as np
from AIPUBuilder.Optimizer.framework import *


@register_plugin(PluginType.Metric, '1.0')
class MaskRcnnCOCOmAPMetric(OptBaseMetric):
    """
    This MaskRcnnCOCOmAPMetric is used for the metric of MaskRcnn model in Optimizer.
    """

    def __init__(self):
        self.AP = []

    def __call__(self, pred, target):
        """
        pred: [bbox, class_ids, score, mask]
        bbox:  [N, (y1, x1, y2, x2)]
        class_ids: [N, *]
        score: [N, *]
        mask:  [N, 1024, 1024, *]
        target: dict{"class_id":'', "box":'', "mask":'}
        """

        batch_detections = torch.cat(
            (
                pred[0],
                pred[4].unsqueeze(dim=2),
                pred[2].unsqueeze(dim=2)
            ),
            dim=2
        ).cpu().numpy()
        batch_mask = pred[1].cpu().numpy()
        assert batch_mask.shape[0] == batch_detections.shape[0], "batch size error in mAPMetric."

        batch_num = batch_mask.shape[0]
        for i in range(batch_num):
            boxes, class_ids, scores, full_masks = self.PostMask(
                batch_detections[i], batch_mask[i])
            ap, precisions, recalls, overlaps = self._ap(
                target['box'][i].cpu().numpy(),
                target['class_id'][i].cpu().numpy(),
                target['mask'][i].cpu().numpy(),
                boxes,
                class_ids,
                scores,
                full_masks
            )
            self.AP.append(ap)

    def reset(self):
        self.AP = []

    def compute(self):
        return np.mean(self.AP)

    def report(self):
        return "mAP @ IoU=50:  %f" % (self.compute())

    def norm_boxes(self, boxes, shape):
        """
        Converts boxes from pixel coordinates to normalized coordinates.
        """
        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.divide((boxes - shift), scale).astype(np.float32)

    def denorm_boxes(self, boxes, shape):
        """
        Converts boxes from normalized coordinates to pixel coordinates.
        """
        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)

    def resize(self, image, output_shape, order=1, mode='constant', cval=0, clip=True,
               preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
        image_3d = torch.unsqueeze(torch.from_numpy(image), 0)
        image_4d = torch.unsqueeze(image_3d, 0)
        resized_img = torch.nn.functional.interpolate(
            image_4d, size=output_shape, mode='bilinear')
        resized_img_2d = torch.squeeze(resized_img)

        return resized_img_2d.cpu().numpy()

    def unmold_mask(self, mask, bbox, image_shape):
        threshold = 0.5
        y1, x1, y2, x2 = bbox
        mask = self.resize(mask, (y2 - y1, x2 - x1))
        mask = np.reshape(mask, (y2 - y1, x2 - x1))
        mask = np.where(mask >= threshold, 1, 0).astype(bool)

        full_mask = np.zeros(image_shape[:2], dtype=bool)
        full_mask[y1:y2, x1:x2] = mask
        return full_mask

    def PostMask(self, detections, mrcnn_mask,
                 original_image_shape=[1024, 1024, 3],
                 image_shape=[1024, 1024, 3],
                 window=[0, 0, 1024, 1024]):

        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]  # shape[N,28,28]

        window = self.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1
        ww = wx2 - wx1
        scale = np.array([wh, ww, wh, ww])

        boxes = np.divide(boxes - shift, scale)
        boxes = self.denorm_boxes(boxes, original_image_shape[:2])

        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        full_masks = []
        for i in range(N):
            full_mask = self.unmold_mask(
                masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + [0, ])

        return boxes, class_ids, scores, full_masks

    def remove_all_zeros_columns(self, x):
        assert len(x.shape) == 2
        return x[~np.all(x == 0, axis=1)]

    def compute_overlaps_masks(self, masks1, masks2):
        """Computes IoU overlaps between two sets of masks.
        masks1, masks2: [Height, Width, instances]
        """
        if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
            return np.zeros((masks1.shape[-1], masks2.shape[-1]))

        masks1 = np.reshape(
            masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
        masks2 = np.reshape(
            masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
        area1 = np.sum(masks1, axis=0)
        area2 = np.sum(masks2, axis=0)

        # IOU
        intersections = np.dot(masks1.T, masks2)
        union = area1[:, None] + area2[None, :] - intersections
        overlaps = intersections / union

        return overlaps

    def find_matches(self,
                     gt_boxes,     gt_class_ids,    gt_masks,
                     pred_boxes, pred_class_ids, pred_scores, pred_masks,
                     iou_threshold=0.5,
                     score_threshold=0.0):
        """
        Finds matches between prediction and ground truth instances.
        """
        gt_boxes = self.remove_all_zeros_columns(gt_boxes)
        gt_masks = gt_masks[..., :gt_boxes.shape[0]]
        pred_boxes = self.remove_all_zeros_columns(pred_boxes)
        pred_scores = pred_scores[:pred_boxes.shape[0]]
        indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[indices]
        pred_class_ids = pred_class_ids[indices]
        pred_scores = pred_scores[indices]
        pred_masks = pred_masks[..., indices]
        overlaps = self.compute_overlaps_masks(pred_masks, gt_masks)
        match_count = 0
        pred_match = -1 * np.ones([pred_boxes.shape[0]])
        gt_match = -1 * np.ones([gt_boxes.shape[0]])
        for i in range(len(pred_boxes)):
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            low_score_idx = np.where(
                overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            for j in sorted_ixs:
                if gt_match[j] > -1:
                    continue
                iou = overlaps[i, j]
                if iou < iou_threshold:
                    break
                if pred_class_ids[i] == gt_class_ids[j]:
                    match_count += 1
                    gt_match[j] = i
                    pred_match[i] = j
                    break

        return gt_match, pred_match, overlaps

    def _ap(self,
            gt_boxes,    gt_class_ids,                gt_masks,
            pred_boxes, pred_class_ids, pred_scores, pred_masks,
            iou_threshold=0.5):
        """
        Compute Average Precision at a set IoU threshold (default 0.5).
        """

        gt_match, pred_match, overlaps = self.find_matches(
            gt_boxes, gt_class_ids, gt_masks,
            pred_boxes, pred_class_ids, pred_scores, pred_masks,
            iou_threshold
        )
        precisions = np.cumsum(pred_match > -1) / \
            (np.arange(len(pred_match)) + 1)
        recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

        precisions = np.concatenate([[0], precisions, [0]])
        recalls = np.concatenate([[0], recalls, [1]])

        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = np.maximum(precisions[i], precisions[i + 1])

        indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
        mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                     precisions[indices])

        return mAP, precisions, recalls, overlaps


@register_plugin(PluginType.Metric, '1.0')
class MaskRcnnTorchmAPMetric(MaskRcnnCOCOmAPMetric):
    """
        This MaskRcnnTorchmAPMetric is used for the metric of MaskRcnn torch model in Optimizer.
        """

    def __init__(self):
        super().__init__()

    def __call__(self, pred, target):

        box_num = pred[5].shape[1]
        batch_num = pred[5].shape[0]
        class_num = pred[6].shape[1]
        boxnum_perclass = pred[6].cpu().numpy().astype(np.int32)  # [1,90]
        class_id = pred[3].cpu().numpy().astype(np.int32)  # [1,90]
        bbox = pred[5].cpu().numpy().astype(np.int32)
        score = pred[7].cpu().numpy()  # [1,100]
        batch_mask = pred[10].cpu().numpy().transpose(0, 1, 3, 4, 2)  # [[1,100,91,28,28]]->[1,100,28,28,91]
        for b in range(batch_num):
            box_class_id = np.zeros([box_num]).astype(np.int32)
            new_box = np.zeros([0, 4]).astype(np.int32)
            new_score = np.zeros([box_num])
            new_mask = np.zeros([box_num, 28, 28])
            full_mask = np.zeros([box_num, 800, 800], dtype=bool)
            offset = 0
            for c in range(class_num):
                cboxnum = int(boxnum_perclass[b, c])
                if cboxnum == 0:
                    continue
                id = int(class_id[b, c])
                new_box = np.concatenate([new_box, bbox[b, offset:offset+cboxnum, :]], axis=0)
                new_mask[offset:offset+cboxnum, ...] = batch_mask[b, offset:offset+cboxnum, :, :, id+1]
                for tmp_offset in range(cboxnum):
                    full_mask[offset+tmp_offset, ...] = self.unmold_mask(
                        new_mask[offset+tmp_offset, ...], new_box[offset+tmp_offset], (800, 800))
                box_class_id[offset:offset+cboxnum] = [id] * cboxnum
                new_score[offset:offset+cboxnum] = score[b, offset:offset+cboxnum]
                offset += cboxnum
            box_class_id = box_class_id[:offset]
            new_score = new_score[:offset]
            full_mask = full_mask[:offset].transpose(1, 2, 0)  # [instance,h,w] -> [h,w,instance]
            ap, precisions, recalls, overlaps = self._ap(
                target['box'][b].cpu().numpy(),
                target['class_id'][b].cpu().numpy(),
                target['mask'][b].cpu().numpy(),
                new_box,
                box_class_id,
                new_score,
                full_mask
            )
            self.AP.append(ap)
