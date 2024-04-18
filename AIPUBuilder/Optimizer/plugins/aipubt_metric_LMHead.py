# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
import torch


@register_plugin(PluginType.Metric, '1.0')
class LMHeadMetric(OptBaseMetric):
    '''
    This plugin calculate correctness rate of Language Model Head output, given
    golden output.

    Sample input: FloatTensor[batch, seq_len, hidden_size]
    Model output: FloatTensor[batch, 1, vocab_size]
    Vocab output: IntTensor[batch, 1]
    Output ref: IntTensor[batch, 1]
    Param: effective_count, vocab output & ref -> [batch, seq_len-effective_count:]
    return (ref == output).sum() / output.size()
    '''

    def __init__(self):
        self.total = 0
        self.correct = 0

    def __call__(self, pred, target):
        vocab = pred[0].argmax(-1)
        self.total += vocab.shape[0]
        self.correct += (vocab.int() == target.argmax(-1).int()).sum().item()

    def reset(self):
        self.correct = 0
        self.total = 0

    def compute(self):
        return self.correct / self.total

    def report(self):
        return f"Correct/Total: {self.compute()}"
