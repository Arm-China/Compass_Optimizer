# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.plugins.aipubt_metric_mAP import BasemAPMetric
import numpy as np
import torch
from collections import defaultdict
from torchvision.ops import nms


@register_plugin(PluginType.Metric, '1.0')
class retinanetmAP(BasemAPMetric):
    """
        https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/retinanet
    """

    def __init__(self, ratios=[1.0, 2.0, 0.5], threshold=0.5, top_n=1000, nms_thres=0.95, detections=300):
        self.ratios = ratios
        self.scales = [4 * 2 ** (i / 3) for i in range(3)]
        self.angles = [-np.pi / 6, 0, np.pi / 6]
        self.anchors = {}
        self.rotated_bbox = None

        self.threshold = threshold
        self.top_n = top_n
        self.nms_thres = nms_thres
        self.detections = detections
        super().__init__()

    def __call__(self, pred, target):
        batch = pred[0].shape[0]

        targets_list = []
        for b in range(batch):
            targets = {}
            for k, v in target.items():
                targets.update({k: v[b].numpy()})
            targets_list.append(targets)

        # Inference post-processing
        decoded = []
        cls_heads = pred[:5]
        box_heads = pred[5:]
        for cls_head, box_head in zip(cls_heads, box_heads):
            # Generate level's anchors
            stride = 640 // cls_head.shape[-1]
            if stride not in self.anchors:
                self.anchors[stride] = self.generate_anchors(stride, self.ratios, self.scales, self.angles)

            # Decode and filter boxes
            decoded.append(self.decode(cls_head, box_head, stride, self.threshold,
                                       self.top_n, self.anchors[stride], self.rotated_bbox))

        for i in range(batch):
            # Perform non-maximum suppression
            decoded = [torch.cat(tensors, 1)[i] for tensors in zip(*decoded)]
            score_list, boxes_list, label_id_list = self.nms(*decoded, self.nms_thres, self.detections)

            predict = defaultdict()
            predict.update({'label_index': label_id_list})
            predict.update({'bbox': boxes_list})
            predict.update({'confidence': score_list})
            predict.update({'image_name': targets_list[i]['image_name']})

            BasemAPMetric.extract_obj_all_class(predict, self.predicts)
            BasemAPMetric.extract_obj_all_class(targets_list[i], self.targets)

    def nms(self, all_scores, all_boxes, all_classes, nms=0.5, ndetections=100):
        'Non Maximum Suppression'

        device = all_scores.device
        out_scores = torch.zeros((ndetections), device=device)
        out_boxes = torch.zeros((ndetections, 4), device=device)
        out_classes = torch.zeros((ndetections), device=device)

        # Per item in batch
        keep = (all_scores.view(-1) > 0).nonzero()
        scores = all_scores[keep].view(-1)
        boxes = all_boxes[keep, :].view(-1, 4)
        classes = all_classes[keep].view(-1)

        if scores.nelement() == 0:
            return np.array([]), np.array([]), np.array([])

        # Sort boxes
        scores, indices = torch.sort(scores, descending=True)
        boxes, classes = boxes[indices], classes[indices]
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1).view(-1)
        keep = torch.ones(scores.nelement(), device=device, dtype=torch.uint8).view(-1)

        for i in range(ndetections):
            if i >= keep.nonzero().nelement() or i >= scores.nelement():
                i -= 1
                break

            # Find overlapping boxes with lower score
            xy1 = torch.max(boxes[:, :2], boxes[i, :2])
            xy2 = torch.min(boxes[:, 2:], boxes[i, 2:])
            inter = torch.prod((xy2 - xy1 + 1).clamp(0), 1)
            criterion = ((scores > scores[i]) |
                         (inter / (areas + areas[i] - inter) <= nms) |
                         (classes != classes[i]))
            criterion[i] = 1

            # Only keep relevant boxes
            scores = scores[criterion.nonzero()].view(-1)
            boxes = boxes[criterion.nonzero(), :].view(-1, 4)
            classes = classes[criterion.nonzero()].view(-1)
            areas = areas[criterion.nonzero()].view(-1)
            keep[(~criterion).nonzero()] = 0

        out_scores[:i + 1] = scores[:i + 1]
        out_boxes[:i + 1, :] = boxes[:i + 1, :]
        out_classes[:i + 1] = classes[:i + 1]

        # Shrink non-zero
        out_scores2, out_boxes2, out_classes2 = [], [], []
        for n in range(ndetections):
            if not torch.equal(out_boxes[n], torch.zeros_like(out_boxes[n])):
                out_scores2.append(out_scores[n].cpu().numpy().tolist())

                out_boxes[n][0], out_boxes[n][1] = out_boxes[n][1] / 480, out_boxes[n][0] / 640
                out_boxes[n][2], out_boxes[n][3] = out_boxes[n][3] / 480, out_boxes[n][2] / 640

                out_boxes2.append(out_boxes[n].cpu().numpy().tolist())
                out_classes2.append(out_classes[n].cpu().numpy().tolist())

        return np.asarray(out_scores2), np.asarray(out_boxes2), np.asarray(out_classes2)

    def decode(self, all_cls_head, all_box_head, stride=1, threshold=0.05, top_n=1000, anchors=None, rotated=False):
        'Box Decoding and Filtering'

        if rotated:
            anchors = anchors[0]
        num_boxes = 4 if not rotated else 6

        device = all_cls_head.device
        anchors = anchors.to(device).type(all_cls_head.type())
        num_anchors = anchors.size()[0] if anchors is not None else 1
        num_classes = all_cls_head.size()[1] // num_anchors
        height, width = all_cls_head.size()[-2:]

        batch_size = all_cls_head.size()[0]
        out_scores = torch.zeros((batch_size, top_n), device=device)
        out_boxes = torch.zeros((batch_size, top_n, num_boxes), device=device)
        out_classes = torch.zeros((batch_size, top_n), device=device)

        # Per item in batch
        for batch in range(batch_size):
            cls_head = all_cls_head[batch, :, :, :].contiguous().view(-1)
            box_head = all_box_head[batch, :, :, :].contiguous().view(-1, num_boxes)

            # Keep scores over threshold
            keep = (cls_head >= threshold).nonzero().view(-1)
            if keep.nelement() == 0:
                continue

            # Gather top elements
            scores = torch.index_select(cls_head, 0, keep)
            scores, indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
            indices = torch.index_select(keep, 0, indices).view(-1)
            classes = (indices / width / height) % num_classes
            classes = classes.type(all_cls_head.type()).to(torch.long)

            # Infer kept bboxes
            x = indices % width
            y = (indices / width) % height
            a = indices / num_classes / height / width
            box_head = box_head.view(num_anchors, num_boxes, height, width)
            boxes = box_head[a.to(torch.long), :, y.to(torch.long), x.to(torch.long)]

            if anchors is not None:
                grid = torch.stack([x.to(torch.long), y.to(torch.long), x.to(torch.long), y.to(torch.long)], 1).type(
                    all_cls_head.type()) * stride + anchors[a.to(torch.long), :]
                boxes = self.delta2box(boxes, grid, [width, height], stride)

            out_scores[batch, :scores.size()[0]] = scores
            out_boxes[batch, :boxes.size()[0], :] = boxes
            out_classes[batch, :classes.size()[0]] = classes

        return out_scores, out_boxes, out_classes

    def delta2box(self, deltas, anchors, size, stride):
        'Convert deltas from anchors to boxes'

        anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
        ctr = anchors[:, :2] + 0.5 * anchors_wh
        pred_ctr = deltas[:, :2] * anchors_wh + ctr
        pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh

        m = torch.zeros([2], device=deltas.device, dtype=deltas.dtype)
        M = (torch.tensor([size], device=deltas.device, dtype=deltas.dtype) * stride - 1)
        def clamp(t): return torch.max(m, torch.min(t, M))
        return torch.cat([
            clamp(pred_ctr - 0.5 * pred_wh),
            clamp(pred_ctr + 0.5 * pred_wh - 1)
        ], 1)

    def generate_anchors(self, stride, ratio_vals, scales_vals, angles_vals=None):
        'Generate anchors coordinates from scales/ratios'

        scales = torch.FloatTensor(scales_vals).repeat(len(ratio_vals), 1)
        scales = scales.transpose(0, 1).contiguous().view(-1, 1)
        ratios = torch.FloatTensor(ratio_vals * len(scales_vals))

        wh = torch.FloatTensor([stride]).repeat(len(ratios), 2)
        ws = torch.sqrt(wh[:, 0] * wh[:, 1] / ratios)
        dwh = torch.stack([ws, ws * ratios], dim=1)
        xy1 = 0.5 * (wh - dwh * scales)
        xy2 = 0.5 * (wh + dwh * scales)
        return torch.cat([xy1, xy2], dim=1)

    def reset(self):
        super().reset()

    def compute(self):
        self.eval_mAP(self.predicts, self.targets, 90, 1)
        return self.mAP

    def report(self):
        return "retinanet_onnx mAP accuracy is %f" % (self.compute())


@register_plugin(PluginType.Metric, '1.0')
class efficientdetmAP(retinanetmAP):
    """
        https://github.com/xuannianz/EfficientDet/tree/master
    """

    def __init__(self, image_size=512, pyramid_levels=[3, 4, 5, 6, 7], ratios=[1.0, 0.5, 2.0], threshold=0.05, top_n=1000, nms_thres=0.5, detections=300):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.ratios = ratios
        self.scales = [2 ** (i / 3) for i in range(3)]

        self.image_size = int(image_size)
        self.anchors = self.generate_anchors([self.image_size, self.image_size])

        self.rotated_bbox = None

        self.threshold = threshold
        self.top_n = top_n
        self.nms_thres = nms_thres
        self.detections = detections

    def __call__(self, pred, target):
        batch = pred[0].shape[0]

        targets_list = []
        for b in range(batch):
            targets = {}
            for k, v in target.items():
                targets.update({k: v[b].numpy()})
            targets_list.append(targets)
        score_list, boxes_list, label_id_list = self.decode(pred[0], pred[1], self.top_n)
        for i in range(batch):
            # Perform non-maximum suppression
            predict = defaultdict()
            predict.update({'label_index': label_id_list[i]})
            predict.update({'bbox': boxes_list[i].numpy()})
            predict.update({'confidence': score_list[i]})
            predict.update({'image_name': targets_list[i]['image_name']})

            BasemAPMetric.extract_obj_all_class(predict, self.predicts)
            BasemAPMetric.extract_obj_all_class(targets_list[i], self.targets)

    def generate_anchors(self, image_shape):
        def anchors_for_shape(base_size=16, ratios=None, scales=None):
            num_anchors = len(ratios) * len(scales)
            anchors = np.zeros((num_anchors, 4))
            # anchors[:, 2:] = base_size * np.tile(np.repeat(scales, len(ratios))[None], (2, 1)).T
            anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
            areas = anchors[:, 2] * anchors[:, 3]
            anchors[:, 2] = np.sqrt(areas / np.tile(ratios, len(scales)))
            anchors[:, 3] = anchors[:, 2] * np.tile(ratios, len(scales))
            anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
            anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
            return anchors

        def shift(shape, stride, anchors):
            shift_x = (np.arange(0, shape[1]) + 0.5) * stride
            shift_y = (np.arange(0, shape[0]) + 0.5) * stride

            shift_x, shift_y = np.meshgrid(shift_x, shift_y)

            shifts = np.vstack((
                shift_x.ravel(), shift_y.ravel(),
                shift_x.ravel(), shift_y.ravel()
            )).transpose()

            shape_0 = anchors.shape[0]
            shape_1 = shifts.shape[0]
            shifts = shifts.reshape((1, shape_1, 4)).transpose((1, 0, 2))
            anchors = anchors.reshape((1, shape_0, 4)) + shifts
            anchors = anchors.reshape((shape_1 * shape_0, 4))
            return anchors

        all_anchors = np.zeros((0, 4), dtype=np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchors = anchors_for_shape(self.sizes[idx], self.ratios, self.scales)
            shape = (np.array(image_shape) + 2 ** p - 1) // (2 ** p)
            shifted_anchors = shift(shape, self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        return np.expand_dims(all_anchors, axis=0)

    def decode(self, box_head, cls_head, top_n=1000):

        batch_size = box_head.size()[0]
        device = cls_head.device

        out_scores = torch.zeros((batch_size, top_n), device=device)
        out_boxes = torch.zeros((batch_size, top_n, 4), device=device)
        out_classes = torch.zeros((batch_size, top_n), device=device)

        for batch in range(batch_size):
            boxes = self.getbox(box_head[batch])[0]
            classes = cls_head[batch].contiguous()
            scores = torch.max(classes, axis=1)[0]
            labels = torch.argmax(classes, axis=1)

            keep = (scores >= self.threshold).nonzero()[:, 0]
            _boxes = torch.index_select(boxes, 0, keep)
            _scores = torch.index_select(scores, 0, keep).type(_boxes.type())

            nms_keep = nms(_boxes, _scores, self.nms_thres)
            keep = torch.index_select(keep, 0, nms_keep)

            # get topk
            _scores = torch.index_select(scores, 0, keep)
            scores, top_indices = torch.topk(_scores, min(top_n, keep.size()[0]), dim=0)
            indices = torch.index_select(keep, 0, top_indices)
            labels = torch.index_select(labels, 0, indices)
            boxes = torch.index_select(boxes, 0, indices)

            out_scores[batch, :scores.size()[0]] = scores
            out_boxes[batch, :boxes.size()[0], :] = boxes / self.image_size
            out_classes[batch, :labels.size()[0]] = labels

        return out_scores, out_boxes, out_classes

    def getbox(self, boxes):
        cxa = (self.anchors[..., 0] + self.anchors[..., 2]) / 2
        cya = (self.anchors[..., 1] + self.anchors[..., 3]) / 2
        wa = self.anchors[..., 2] - self.anchors[..., 0]
        ha = self.anchors[..., 3] - self.anchors[..., 1]
        ty, tx, th, tw = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]

        w = torch.exp(tw) * wa
        h = torch.exp(th) * ha
        cy = ty * ha + cya
        cx = tx * wa + cxa
        ymin = torch.clamp(cy - h / 2., 0, self.image_size)
        xmin = torch.clamp(cx - w / 2., 0, self.image_size)
        ymax = torch.clamp(cy + h / 2., 0, self.image_size)
        xmax = torch.clamp(cx + w / 2., 0, self.image_size)
        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)

    def report(self):
        return "mAP accuracy is %f" % (self.compute())
