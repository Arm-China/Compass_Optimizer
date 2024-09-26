# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import editdistance
import numpy as np


'''
WER: word error rate
Work Error Rate =  100 * (insertions + substitutions + deletions) / (total words in correct transcript)
'''


@register_plugin(PluginType.Metric, '1.0')
class WERMetric(OptBaseMetric):
    """
    This WERMetric is used for the metric of deepspeech_official/wavenet models in Optimizer.

    Word error rate (WER) is a common metric of the performance of a speech recognition or machine translation system.
    Work Error Rate =  100 * (insertions + substitutions + deletions) / (total words in correct transcript)
    """

    def __init__(self, EOF=''):
        self.predictions = []
        self.WER = 0
        self.EOF = EOF

    def __call__(self, preds, targets):
        '''
        :param preds:
        :param targets: list(padding_label, act_label_len), padding_label.shape=(batch_size, padding_len), act_label_len.shape=(batch_size, act_len)
        :return:
        '''
        preds = preds[0].cpu().numpy()
        padded_targets = targets[0].cpu().numpy()
        act_len = targets[1].cpu().numpy()
        targets = padded_targets
        for i in range(targets.shape[0]):
            flatten_pred = preds[i].reshape([-1])
            eof_value = int(self.EOF) if len(self.EOF) > 0 else flatten_pred[-1]
            flatten_pred = flatten_pred[flatten_pred != eof_value]
            flatten_target = targets[i][:act_len[i]].reshape([-1])
            self.predictions.append(editdistance.eval(flatten_pred, flatten_target) / len(flatten_target))

    def reset(self):
        self.predictions = []
        self.WER = 0

    def compute(self):
        self.WER = np.average(np.array(self.predictions))
        return self.WER

    def report(self):
        return "ASR Word Error Rate(WER) is %f" % (self.compute())
