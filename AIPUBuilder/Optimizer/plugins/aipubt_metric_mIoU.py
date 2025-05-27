# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

import torch
import cv2
import numpy as np


@register_plugin(PluginType.Metric, '1.0')
class mIoUMetricBase(OptBaseMetric):
    def __init__(self):
        self.confusion_matrix = None
        self.label = None
        self.class_num = 1
        self.epsilon = 0.000000000001
        self.channel_axis = 3
        self.perm = None

    def __call__(self, pred, target):
        '''
        we assume the channel_axis of target is 3, so we only need to transpose the prediction.
        '''
        prediction = torch.argmax(pred[0], dim=self.channel_axis, keepdim=True)
        if self.confusion_matrix is None:
            self.class_num = pred[0].shape[self.channel_axis]
            self.confusion_matrix = np.zeros((self.class_num, self.class_num))
            self.label = np.arange(self.class_num)
            p_dim_idx = [d for d in range(prediction.dim())]
            self.perm = p_dim_idx[:self.channel_axis] + p_dim_idx[self.channel_axis+1:] + [self.channel_axis]
        if len(prediction.shape) != len(target.shape):
            target = target.view(prediction.shape)
        prediction = prediction.cpu().numpy().transpose(self.perm)
        target = target.cpu().numpy()
        for n in range(target.shape[0]):
            g = target[n].flatten()
            p = prediction[n].flatten()
            mask = (g >= 0) & (g < self.class_num)
            self.confusion_matrix += np.bincount(self.class_num * g[mask].astype(int) + p[mask],
                                                 minlength=self.class_num ** 2).reshape(self.class_num, self.class_num)

    def reset(self):
        self.confusion_matrix = None
        self.label = None
        self.class_num = 1

    def compute(self):
        iou = np.diag(self.confusion_matrix) / \
            (self.confusion_matrix.sum(axis=0) + self.confusion_matrix.sum(axis=1) -
             np.diag(self.confusion_matrix) + self.epsilon)
        return np.mean(iou)

    def report(self):
        return "mIoU is %f" % (self.compute())


@register_plugin(PluginType.Metric, '1.0')
class mIoUMetric(mIoUMetricBase):
    """
    This mIoUMetric is used for the metric of deeplab_tflite/deeplab/fcn/enet/erfnet models in Optimizer.
    This plugin computes the mean Intersection-Over-Union metric and assume the class num in channel axis = 3.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, pred, target):
        super().__call__(pred, target)

    def reset(self):
        super().reset()

    def compute(self):
        return super().compute()

    def report(self):
        return "mIoU is %f" % (self.compute())


@register_plugin(PluginType.Metric, '1.0')
class mIoUNCHWMetric(mIoUMetricBase):
    """
    This mIoUNCHWMetric is used for the metric of icnet_caffe/fcn_caffe/enet_caffe/erfnet models in Optimizer.
    This plugin computes the mean Intersection-Over-Union metric and assume the class num in channel axis = 1.
    """

    def __init__(self):
        super().__init__()
        self.channel_axis = 1

    def __call__(self, pred, target):
        super().__call__(pred, target)

    def reset(self):
        super().reset()

    def compute(self):
        return super().compute()

    def report(self):
        return "mIoU is %f" % (self.compute())


@register_plugin(PluginType.Metric, '1.0')
class mIoUafterargmaxMetric(mIoUMetricBase):
    """
    This mIoUafterargmaxMetric is used for the metric of bisenet_tensorflow/onnx_ann/onnx_apcnet model in Optimizer.
    This plugin computes the mean Intersection-Over-Union metric.
    """

    def __init__(self, class_num=19, layout='NHWC'):
        super().__init__()
        self.class_num = int(class_num)
        self.layout = layout

    def __call__(self, pred, target):
        if len(pred[0].shape) == 3:
            prediction = pred[0][..., np.newaxis]
        else:
            prediction = pred[0]
        if self.confusion_matrix is None:
            self.confusion_matrix = np.zeros((self.class_num, self.class_num))
            self.label = np.arange(self.class_num)
        if len(prediction.shape) != len(target.shape):
            target = target.view(prediction.shape)
        prediction = prediction.cpu().numpy()
        target = target.cpu().numpy()
        if self.layout == 'NHWC':
            prediction = prediction.transpose((0, 3, 1, 2))
            target = target.transpose((0, 3, 1, 2))
        for n in range(target.shape[0]):
            g = target[n].flatten()
            p = prediction[n].flatten()
            mask = (g >= 0) & (g < self.class_num)
            self.confusion_matrix += np.bincount(self.class_num * g[mask].astype(int) + p[mask].astype(int),
                                                 minlength=self.class_num ** 2).reshape(self.class_num, self.class_num)

    def reset(self):
        self.confusion_matrix = None
        self.label = None

    def compute(self):
        return super().compute()

    def report(self):
        return "mIoU is %f" % (self.compute())


@register_plugin(PluginType.Metric, '1.0')
class mIoUDilation8Metric(mIoUMetricBase):
    """
    This mIoUDilation8Metric is used for the metric of dilation_8 model in Optimizer.
    This plugin computes the mean Intersection-Over-Union metric.
    """

    def __init__(self, class_num=21, width=500, height=500):
        super().__init__()
        self.class_num = int(class_num)
        self.width = width
        self.height = height
        self.plette = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [
                           64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]  # 21 class

    def __call__(self, pred, target):

        def interp_label(prob, shapes, width, height):  # require NCHW
            pred = np.zeros((prob.shape[0], 1, height, width))
            prob = torch.argmax(prob, dim=1, keepdim=True).cpu().numpy()
            for n in range(prob.shape[0]):
                if shapes[n][0] >= shapes[n][1]:  # H>W
                    real_w = int(shapes[n][1]/shapes[n][0] * prob.shape[3])
                    local_prob = prob[n, :, :, :real_w]
                elif shapes[n][1] > shapes[n][0]:  # W>H
                    real_h = int(shapes[n][0]/shapes[n][1] * prob.shape[2])
                    local_prob = prob[n, :, :real_h, :]

                # squared region
                square_shape = max(local_prob.shape[1], local_prob.shape[2])
                local_prob_square = np.zeros((1, square_shape, square_shape))
                offset_h = (square_shape-local_prob.shape[1])//2
                offset_w = (square_shape-local_prob.shape[2])//2
                local_prob_square[:, offset_h:offset_h+local_prob.shape[1],
                                  offset_w:offset_w+local_prob.shape[2]] = local_prob

                pred[n] = cv2.resize(local_prob_square.transpose(
                    1, 2, 0), (height, width), interpolation=cv2.INTER_NEAREST)
            return pred

        if len(pred[0].shape) == 3:
            prediction = pred[0][..., np.newaxis]
        else:
            prediction = pred[0]
        prediction = interp_label(prediction, target[1], self.width, self.height).transpose(0, 2, 3, 1)
        if self.confusion_matrix is None:
            self.confusion_matrix = np.zeros((self.class_num, self.class_num))
            self.label = np.arange(self.class_num)
        target = target[0]
        if len(prediction.shape) != len(target[0].shape):
            target = target.view(prediction.shape)
        prediction = prediction.transpose((0, 3, 1, 2)).round()
        target = target.cpu().numpy().transpose((0, 3, 1, 2))
        for n in range(target.shape[0]):  # [500,500,1] vs [500,500,1]
            g = target[n].flatten()
            p = prediction[n].flatten()
            mask = (g >= 0) & (g < self.class_num)
            self.confusion_matrix += np.bincount(self.class_num * g[mask].astype(int) + p[mask].astype(int),
                                                 minlength=self.class_num ** 2).reshape(self.class_num, self.class_num)

    def reset(self):
        self.confusion_matrix = None
        self.label = None

    def compute(self):
        return super().compute()

    def report(self):
        return "mIoU is %f" % (self.compute())


@register_plugin(PluginType.Metric, '1.0')
class mIoUpointnetMetric(mIoUMetricBase):
    """
    This mIoUpointnetMetric is used for the metric of pointnet model in Optimizer.
    This plugin computes the mean Intersection-Over-Union metric.
    """

    def __init__(self,
                 scene_point_index='scene_point_index.npy',
                 scene_sample_wight='scene_smpw.npy',
                 class_num=13, total_block=970, label_size=1272342, num_point=4096):
        super().__init__()
        self.class_num = int(class_num)
        self.batch_size = 32
        self.total_block = total_block
        self.local_batch = 0
        self.label_size = label_size
        self.num_point = num_point
        self.vote_list = np.zeros((self.label_size, self.class_num))
        self.batch_point_index = np.zeros((self.batch_size, self.num_point))
        self.batch_sample_wight = np.zeros((self.batch_size, self.num_point))
        self.scene_point_index = np.load(scene_point_index)
        self.scene_sample_wight = np.load(scene_sample_wight)
        self.target = None

    def __call__(self, pred, target):
        def vote_pool(vote_list, point_idx, pred_label, weight):
            batch_size = pred_label.shape[0]
            num_size = pred_label.shape[1]
            for b_idx in range(batch_size):
                for n_idx in range(num_size):
                    if weight[b_idx, n_idx] != 0 and not np.isinf(weight[b_idx, n_idx]):
                        self.vote_list[int(point_idx[b_idx, n_idx]), int(pred_label[b_idx, n_idx])] += 1
            return vote_list

        batch_pred_label = pred[0].contiguous().cpu().data.max(2)[1]

        start_idx = self.local_batch * self.batch_size
        end_idx = min((self.local_batch + 1) * self.batch_size, self.total_block)
        real_batch_size = end_idx - start_idx

        self.batch_point_index[0:real_batch_size, ...] = self.scene_point_index[start_idx:end_idx, ...]
        self.batch_sample_wight[0:real_batch_size, ...] = self.scene_sample_wight[start_idx:end_idx, ...]

        self.vote_list = vote_pool(self.vote_list, self.batch_point_index[0:real_batch_size, ...],
                                   batch_pred_label[0:real_batch_size, ...], self.batch_sample_wight[0:real_batch_size, ...])
        self.local_batch += 1
        if self.target == None:
            self.target = target[0]

    def reset(self):
        self.iou_map = None
        self.label = None
        self.local_batch = 0

    def compute(self):
        local_target = self.target.reshape(-1, self.label_size)[0].cpu().numpy()
        self.pred_label = np.argmax(self.vote_list, 1)
        total_seen_class_tmp = [0 for _ in range(self.class_num)]
        total_correct_class_tmp = [0 for _ in range(self.class_num)]
        total_iou_deno_class_tmp = [0 for _ in range(self.class_num)]
        for l in range(self.class_num):
            total_seen_class_tmp[l] += np.sum((local_target == l))
            total_correct_class_tmp[l] += np.sum((self.pred_label == l) & (local_target == l))
            total_iou_deno_class_tmp[l] += np.sum(((self.pred_label == l) | (local_target == l)))

        iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=float) + 1e-6)
        arr = np.array(total_seen_class_tmp)
        return np.mean(iou_map[arr != 0])

    def report(self):
        return "mIoU is %f" % (self.compute())


@register_plugin(PluginType.Metric, '1.0')
class mIoUwithresizeMetric(mIoUMetricBase):
    """
    This mIoUMetric is used for the metric of deeplab_onnx models in Optimizer.
    This plugin computes the mean Intersection-Over-Union metric
    """

    def __init__(self, class_num=21, size=513, channel_axis=1):
        self.label = None
        self.class_num = class_num
        self.size = size
        self.epsilon = 0.000000000001
        self.channel_axis = channel_axis
        self.hist = np.zeros((self.class_num, self.class_num))

    def __call__(self, pred, target):
        prediction = torch.nn.functional.interpolate(pred[0], size=(
            self.size, self.size), mode="bilinear", align_corners=False)
        prediction = torch.nn.functional.softmax(prediction, dim=self.channel_axis)
        prediction = torch.argmax(prediction, dim=self.channel_axis, keepdim=True)
        prediction = prediction.cpu().numpy()
        target = target.cpu().numpy()
        for n in range(target.shape[0]):
            g = target[n].flatten()
            p = prediction[n].flatten()
            mask = (g >= 0) & (g < self.class_num)
            hist = np.bincount(
                self.class_num * g[mask].astype(int) + p[mask],
                minlength=self.class_num ** 2,
            ).reshape(self.class_num, self.class_num)
            self.hist += hist

    def reset(self):
        self.label = None
        self.hist = np.zeros((self.class_num, self.class_num))

    def compute(self):
        hist = self.hist
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        valid = hist.sum(axis=1) > 0  # added
        mean_iu = np.nanmean(iu[valid])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.class_num), iu))
        return mean_iu

    def report(self):
        return "mIoU is %f" % (self.compute())
