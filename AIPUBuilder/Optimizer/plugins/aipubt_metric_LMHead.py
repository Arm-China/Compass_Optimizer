# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
import torch


@register_plugin(PluginType.Metric, '1.0')
class LMHeadMetric(OptBaseMetric):
    '''
    Compare logits PPL with label
    CrossEntropyLoss([batch, vocab_size], [batch, 1(token id)]) -> [batch, 1(neg log liklihood)]
    PPL = exp([batch, 1(nll)].mean())
    '''

    def __init__(self):
        self.nlls = []
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, pred, target):
        vocab = pred[0][:, -1, :]  # [batch, seqlen, vocabsize] -> [batch, vocabsize]
        nll = self.loss(vocab, target[0][:, 0])
        self.nlls.append(nll)

    def reset(self):
        self.nlls = []

    def compute(self):
        total_nll = torch.tensor(self.nlls)
        return torch.exp(total_nll.mean())

    def report(self):
        return f"Correct/Total: {self.compute()}"
