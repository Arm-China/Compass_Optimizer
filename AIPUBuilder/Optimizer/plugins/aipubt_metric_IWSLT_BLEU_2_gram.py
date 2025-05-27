# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.logger import OPT_INFO
from AIPUBuilder.Optimizer.framework import *

import os
import codecs
import tempfile
import re


@register_plugin(PluginType.Metric, '0.01')
class IWSLTBLEU2gramMetric(OptBaseMetric):
    """
    This IWSLTBLEU2gramMetric is used for the metric of transformer_mini_tensorflow model in Optimizer.
    """

    def __init__(self, perf_fname='./multi_bleu.perl'):
        self.reset()
        # wget from https://github.com/Kyubyong/transformer/blob/master/multi-bleu.perl
        self.perl_fname = perf_fname

    def __call__(self, pred, target):
        hypotheses = pred[0]
        self.lines_num = int(target['lines'][0].item())
        self.vocab_fpath = target['vocab'][0]
        self.gt_fpath = target['gt'][0]
        batch_size = hypotheses.shape[0]
        for i in range(batch_size):
            self.pt_lines.append(hypotheses[i].cpu().numpy())

    def reset(self):
        self.vocab_fpath = ''
        self.gt_fpath = ''
        self.pt_lines = []
        self.lines_num = 0

    def compute(self):
        vocab = [line.split()[0] for line in codecs.open(
            self.vocab_fpath, encoding='utf-8').read().splitlines()]
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}

        _hypotheses = []
        for h in self.pt_lines:
            sent = "".join(idx2token[idx] for idx in h)
            sent = sent.split("</s>")[0].strip()
            sent = sent.replace("▁", " ")
            _hypotheses.append(sent.strip())
        hypotheses = _hypotheses[:self.lines_num]

        score = float('nan')

        _, pt_fname = tempfile.mkstemp(text=True)
        with open(pt_fname, 'w') as pt_fout:
            OPT_INFO('the predicted content is in file: ' + pt_fname)
            pt_fout.write("\n".join(hypotheses))
            pt_fout.flush()
            # wget from https://github.com/Kyubyong/transformer/blob/master/multi-bleu.perl
            perl_fname = self.perl_fname
            score_fname = pt_fname + '.score'
            with codecs.open(self.gt_fpath, 'r', encoding='utf-8') as gt_f:
                gt_lines = gt_f.readlines()[:self.lines_num]
                gt_fname = pt_fname + '.gt'
                with codecs.open(gt_fname, 'w', encoding='utf-8') as gt_fout:
                    gt_fout.writelines(gt_lines)
                    gt_fout.flush()
                    OPT_INFO('the groundtruth content is in file: ' + gt_fname)
                    get_bleu_score = "perl {} {} < {} > {}".format(
                        perl_fname, gt_fname, pt_fname, score_fname)
                    OPT_INFO('the get bleu cmd is : ' + get_bleu_score)
                    os.system(get_bleu_score)
                    with open(score_fname, 'r') as score_f:
                        OPT_INFO(
                            'the bleu score content is in file: ' + score_fname)
                        bleu_score_report = score_f.read()
                        OPT_INFO(bleu_score_report)
                        try:
                            score = re.findall(
                                "BLEU = ([^,]+)", bleu_score_report)[0]
                            score = float(score)
                        except:
                            pass

        return score

    def report(self):
        return "BLEU score is %f" % (self.compute())
