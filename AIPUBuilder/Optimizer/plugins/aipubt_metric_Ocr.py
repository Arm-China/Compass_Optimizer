# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
import torch
import numpy as np
import os


class BaseRecLabelDecode(object):
    def __init__(self,
                 character_txt_path=None,
                 character_type='ch',
                 use_space_char=False):
        self.character_txt_path = character_txt_path
        self.character_type = character_type.lower()
        self.use_space_char = use_space_char
        self.character_set = self.generate_character_dict()

    def generate_character_dict(self):
        if self.character_type == "en":
            character_set = "0123456789abcdefghijklmnopqrstuvwxyz"
        else:
            character_set = []
            assert self.character_txt_path is not None, "character_txt_path should not be None and character_type is {}".format(
                self.character_type)
            with open(self.character_txt_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    character_set.append(line)
            if self.use_space_char:
                character_set.append(" ")
        dict_character = ['blank'] + character_set
        return dict_character

    # Convert text-index to text
    def decode_index(self, text_index, text_prob=None, remove_duplicate_character=False):
        result_list = []
        invalid_index = [0]  # index 0 means invalid character
        batch_size = text_index.shape[0]

        for batch_idx in range(batch_size):
            character_list = []
            confidence_list = []
            for idx in range(len(text_index[batch_idx])):
                max_index = (text_index[batch_idx][idx]).int().item()
                if max_index in invalid_index:
                    continue
                if remove_duplicate_character:
                    if idx > 0 and (text_index[batch_idx][idx - 1]).int().item() == max_index:
                        continue
                character = self.character_set[max_index]
                character_list.append(character)
                confidence = 1 if text_prob is None else max_index
                confidence_list.append(confidence)
            text = ''.join(character_list)
            mean_confidence = np.mean(confidence_list)
            result_list.append((text, mean_confidence))
        return result_list


class OcrDecode(BaseRecLabelDecode):
    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False):
        super(OcrDecode, self).__init__(character_dict_path,
                                        character_type, use_space_char)

    def __call__(self, preds, label=None):
        preds_prob, preds_idx = torch.max(preds, dim=2)
        text = self.decode_index(preds_idx, preds_prob, remove_duplicate_character=True)

        if label is None:
            return text

        label = self.decode_index(label)
        return text, label


@register_plugin(PluginType.Metric, '1.0')
class OcrMetric(OptBaseMetric):
    """
    This OcrMetric is used for the metric of Optical Character Recognition models.
    """

    def __init__(self, character_type='ch',
                 character_dict_path=None,
                 use_space_char='true'):
        self.correct = 0
        self.total = 0
        if use_space_char.lower() == 'true':
            use_space_char = True
        elif use_space_char.lower() == 'false':
            use_space_char = False
        else:
            use_space_char = True
            OPT_WARN('the third argument of OcrMetricdoes must be True or False, now is %s, use default value(True) instead of this' % use_space_char)
        self.decoder = OcrDecode(
            character_dict_path, character_type, use_space_char)

    def __call__(self, pred, target):
        data_array = pred[0]
        batch = len(target)
        rec_result = self.decoder(data_array)
        for batch_idx in range(batch):
            if rec_result[batch_idx][0] == target[batch_idx]:
                self.correct += 1
        self.total += batch

    def reset(self):
        self.correct = 0
        self.total = 0

    def compute(self):
        try:
            acc = float(self.correct) / float(self.total)
            return acc
        except ZeroDivisionError:
            OPT_ERROR('zeroDivisionError: Topk acc total label = 0')
            return float("-inf")

    def report(self):
        return "ocr accuracy is %f" % (self.compute())
