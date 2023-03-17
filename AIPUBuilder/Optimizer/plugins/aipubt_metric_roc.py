# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import numpy as np
import torch


@register_plugin(PluginType.Metric, '1.0')
class ROCMetric(OptBaseMetric):
    """
    This ROCMetric is used for the metric of arcface_onnx model in Optimizer.
    """

    def __init__(self, folds_num=10):
        self.ret = []
        self.label = []
        self.folds_num = folds_num

    def __call__(self, pred, target):
        for data, label in zip(pred[0].cpu().numpy(), target.cpu().numpy()):
            self.ret.append(data)
            self.label.append(label)

    def reset(self):
        self.correct = 0
        self.total = 0

    def compute(self):
        num = len(self.ret)
        label = np.array(self.label[:num // 4])
        thres_list = np.arange(0, 4, 0.01)
        ebd_list = [
            np.array(self.ret[:num // 2]),
            np.array(self.ret[num // 2:])]
        ebd = ebd_list[0].copy()
        ebd = ebd / torch.linalg.norm(torch.Tensor(ebd), ord=2,
                                      axis=1, keepdim=True).cpu().numpy()
        ebd = ebd_list[0] + ebd_list[1]
        ebd = ebd / torch.linalg.norm(torch.Tensor(ebd), ord=2,
                                      axis=1, keepdim=True).cpu().numpy()
        ebd0 = ebd[0::2]
        ebd1 = ebd[1::2]

        nums = min(len(label), ebd0.shape[0])
        acc = []
        diff = np.subtract(ebd0, ebd1)
        dist = np.sum(np.square(diff), 1)
        # cal roc
        for train, test in self.KFold(nums, self.folds_num):
            acc_train = []
            for thres in thres_list:
                acc_train.append(self.get_acc(thres, dist[train], label[train]))
            index = np.argmax(acc_train)
            acc.append(self.get_acc(thres_list[index], dist[test], label[test]))

        return np.mean(acc)

    @staticmethod
    def KFold(n=6000, n_splits=10):
        indices = np.arange(n)
        o_len = n // n_splits
        remain_num = n % n_splits
        case_num = np.array([o_len] * n_splits)
        case_num[:remain_num] = case_num[:remain_num] + 1
        for idx, num in enumerate(case_num):
            index = np.sum(case_num[:idx])
            test = indices[index: index + num]
            train = [i for i in indices if i not in test]
            yield train, test

    @staticmethod
    def get_acc(thres, distance, label):
        mask = distance < thres
        acc = float(np.count_nonzero(mask == label))/distance.size
        return acc

    def report(self):
        return "accuracy is %f" % (self.compute())
