# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch.nn.functional as func
import numpy as np


@register_plugin(PluginType.Metric, '1.0')
class LanenetMetric(OptBaseMetric):
    """
    This LanenetMetric is used for the metric of poly lanenet model in Optimizer.
    The input image size of lanenet model is 360x640(original image is 720x1280)
    """

    def __init__(self):
        # lstsq replace sklearn linear_regression fit
        self.lr = np.linalg.lstsq
        self.pixel_thresh = 20
        self.pt_thresh = 0.85
        self.image_size = [720, 1280]
        self.lanes_pred = []
        self.lanes_gt = []
        self.conf_threshold = 0.5

    def calc_angle(self, xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            k = self.lr(np.c_[np.ones((ys.shape[0], 1)), ys], xs)[0][1]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    def calc_line_accuracy(self, pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    def decode(self, all_outputs, conf_threshold=0.5):
        # outputs are reshaped to score + upper + lower + 4 coeffs
        outputs = all_outputs.reshape(-1, 7)
        outputs[:, 1] = outputs[0, 1]
        outputs[:, 0] = func.sigmoid(outputs[:, 0])
        outputs[outputs[:, 0] < conf_threshold] = 0

        return outputs

    def __call__(self, pred, target):
        batch_size = pred[0].shape[0]

        for batch in range(batch_size):
            all_outputs = pred[0][batch:batch+1]

            outputs = self.decode(all_outputs, self.conf_threshold).cpu().numpy()
            y_samples = target['h_samples'][0].cpu().numpy()
            # prediction result convert to lanes
            ys = np.array(y_samples) / self.image_size[0]
            pred_lane = []
            for lane in outputs:
                if lane[0] == 0:
                    continue
                lane_pred = np.polyval(lane[3:], ys) * self.image_size[1]
                lane_pred[(ys < lane[1]) | (ys > lane[2])] = -2
                pred_lane.append(list(lane_pred))

            self.lanes_pred.append(pred_lane)
            self.lanes_gt.append([target['lanes'].cpu().numpy(), y_samples])

    def one_lane_acc(self, pred, gt, y_samples):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')

        angles = [self.calc_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [self.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs):
            accs = [self.calc_line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.

            if max_acc < self.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.), 1.)

    def lanes_acc(self, lanes_pred, lanes_gt):

        if len(lanes_pred) != len(lanes_gt):
            raise Exception('the predictions number of all the test not equtal to gt number')
        accuracy, fp, fn = 0., 0., 0.

        for idx in range(len(lanes_pred)):

            pred_lanes = lanes_pred[idx]

            gt_lanes = lanes_gt[idx][0]
            y_samples = lanes_gt[idx][1]
            try:
                a, p, n = self.one_lane_acc(pred_lanes, gt_lanes[0], y_samples)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(lanes_gt)
        return accuracy/num, fp/num, fn/num

    def reset(self):
        pass

    def compute(self):
        return self.lanes_acc(self.lanes_pred, self.lanes_gt)

    def report(self):
        acc, fp, fn = self.compute()
        return ("lane Acc is %f FP is %f FN is %f" % (acc, fp, fn))
