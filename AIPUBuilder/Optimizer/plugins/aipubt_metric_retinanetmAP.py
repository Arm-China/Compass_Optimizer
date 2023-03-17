# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.plugins.aipubt_metric_mAP import BasemAPMetric
import numpy as np
import torch
from collections import defaultdict


@register_plugin(PluginType.Metric, '1.0')
class retinanetmAP(BasemAPMetric):
    """
        https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/retinanet
    """

    def __init__(self, *args):
        self.ratios = [1.0, 2.0, 0.5]
        self.scales = [4 * 2 ** (i / 3) for i in range(3)]
        self.angles = [-np.pi / 6, 0, np.pi / 6]
        self.anchors = {}
        self.rotated_bbox = None

        self.threshold = 0.5
        self.top_n = 1000
        self.nms_thres = 0.95
        self.detections = 300
        super().__init__(*args)

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
