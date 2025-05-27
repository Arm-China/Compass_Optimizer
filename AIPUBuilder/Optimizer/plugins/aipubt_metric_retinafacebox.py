# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch
from math import ceil
from itertools import product as product
import numpy as np

from AIPUBuilder.Optimizer.plugins.aipubt_metric_widerface import WiderFaceMetric


@register_plugin(PluginType.Metric, '1.0')
class RetinafaceboxMetric(WiderFaceMetric):
    """
    This RetinafaceboxMetric is used for the metric of RetinaFace_onnx model in Optimizer.
    The input image size of facebox model is 640x640.
    """

    def __init__(self):
        self.cfg = {
            'name': 'retinaface_mobilnet0.25',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': True,
            'batch_size': 32,
            'ngpu': 1,
            'epoch': 250,
            'decay1': 190,
            'decay2': 220,
            'image_size': 640,
            'pretrain': True,
            'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
            'in_channel': 32,
            'out_channel': 64
        }
        self.image_size = [self.cfg['image_size'], self.cfg['image_size']]
        self.scale = torch.Tensor([self.image_size[1], self.image_size[0], self.image_size[1], self.image_size[0]])
        self.scale1 = torch.Tensor(self.image_size * 5)
        self.confidence_threshold = 0.02
        self.nms_threshold = 0.4
        self.topk = 5000
        self.box_pred = []
        self.box_gt = []
        self.iou_thresh = 0.5

    def __call__(self, pred, target):
        resize = 1  # this can change in github code when origin_size == False
        batch_size = pred[0].shape[0]
        for batch in range(batch_size):
            delta, landms, conf, = pred[0][batch:batch+1].cpu(), pred[1][batch:batch +
                                                                         1].cpu(), pred[2][batch:batch+1].cpu()
            anchor_data = self.anchors_data().data
            boxes = self.decode(delta.data.squeeze(0), anchor_data, self.cfg['variance'])
            boxes = boxes * self.scale
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

            # ignore low scores
            inds = np.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = list(scores.argsort()[::-1][:self.topk])
            boxes = boxes[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = self.nms(dets)
            dets = dets[keep, :]

            boxes = []
            for box in target['bbox'][batch].cpu().numpy():
                boxes.append(box)
            self.box_gt.append([boxes,
                                target['easy'][batch].cpu().numpy(),
                                target['medium'][batch].cpu().numpy(),
                                target['hard'][batch].cpu().numpy()])
            self.box_pred.append(dets)

    def decode(self, delta, anchors, variances):

        boxes = torch.cat((
            anchors[:, :2] + delta[:, :2] * variances[0] * anchors[:, 2:],
            anchors[:, 2:] * torch.exp(delta[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def anchors_data(self):
        anchors = []
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)]
                             for step in self.cfg['steps']]
        self.steps = self.cfg['steps']
        self.clip = self.cfg['clip']
        all_min_sizes = self.cfg['min_sizes']
        for k, f in enumerate(self.feature_maps):
            min_sizes = all_min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

    def reset(self):
        pass

    def compute(self):
        return self.eval_map(self.box_pred, self.box_gt)

    def report(self):
        aps = self.compute()
        return ("mAP accuracy easy is %f medium is %f hard is %f" % (aps[0], aps[1], aps[2]))
