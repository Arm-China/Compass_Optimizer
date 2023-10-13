# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
from itertools import product as product
import numpy as np


@register_plugin(PluginType.Metric, '1.0')
class SphereFaceMetric(OptBaseMetric):
    """
    This SphereFaceMetric is used for the metric of sphereface model in Optimizer.
    The input image size of centerface model is 112x96.
    https://github.com/wy1iu/sphereface/blob/master/README.md
    """

    def __init__(self):

        self.predicts = []
        self.acc_mean = None
        self.acc_std = None
        self.acc_thd_mean = None

    def __call__(self, pred, target):
        batch_size = pred[0].shape[0]
        # TODO: now defaults metric batch size = 4
        if batch_size != 4:
            OPT_WARN(f"now Spherefacemetric is used for metric_batch_size=4")

        f1, f2 = pred[0][:1, ...].reshape([-1]), pred[0][2:3, ...].reshape([-1])
        cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)

        name1 = target[0]['name1'][0]
        name2 = target[1]['name2'][0]
        sameflag = target[2]['sameflag'][0]
        self.predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, cosdistance, sameflag))

    def reset(self):
        self.predicts = []
        self.acc_mean = None
        self.acc_std = None
        self.acc_thd_mean = None

    def compute(self):
        accuracy = []
        thd = []
        thres_list = np.arange(-1.0, 1.0, 0.005)
        predicts = np.array(list(map(lambda line: line.strip().split(), self.predicts)))
        for (train, test) in self.KFold(n=6000, n_folds=10):
            acc_list = []
            for thres in thres_list:
                acc_list.append(self.eval_acc(thres, predicts[train]))
            index = np.argmax(acc_list)
            best_thresh = thres_list[index]
            accuracy.append(self.eval_acc(best_thresh, predicts[test]))
            thd.append(best_thresh)
        self.acc_mean = np.mean(accuracy)
        self.acc_std = np.std(accuracy)
        self.acc_thd_mean = np.mean(thd)
        return self.acc_mean, self.acc_std, self.acc_thd_mean

    def report(self):
        aps = self.compute()
        return ("sphereface accuracy: mean is %f, std is %f,  mean of best threshold is %f." % (aps[0], aps[1], aps[2]))

    @staticmethod
    def KFold(n=6000, n_folds=10):
        indices = np.arange(n)
        o_len = n // n_folds
        remain_num = n % n_folds
        case_num = np.array([o_len] * n_folds)
        case_num[:remain_num] = case_num[:remain_num] + 1
        for idx, num in enumerate(case_num):
            index = np.sum(case_num[:idx])
            test = indices[index: index + num]
            train = [i for i in indices if i not in test]
            yield train, test

    @staticmethod
    def eval_acc(threshold, diff):
        target = []
        pred = []
        for d in diff:
            same = 1 if float(d[2]) > threshold else 0
            pred.append(same)
            target.append(int(d[3]))
        target = np.array(target)
        pred = np.array(pred)
        accuracy = 1.0*np.count_nonzero(target == pred)/len(target)
        return accuracy
