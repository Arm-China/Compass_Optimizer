# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import numpy as np
import torch

from AIPUBuilder.Optimizer.framework import *

from torch.utils.data import Dataset
from collections import defaultdict


@register_plugin(PluginType.Dataset, '1.0')
class widerfaceDataset(Dataset):
    """
    This widerfaceDataset is used for the centerface model in Optimizer.
    """

    def __init__(self, data_file=None, label_file=None):
        '''
        :param data_file: ndarray in npy file.
        :param label_file: a dict format in .npy file and format is {idx: [bbox, easy, medium, hard]}
                           idx: int,
                           bbox: ndarray, which means box coordination,
                           easy: list,
                           medium: list,
                           hard: list
        '''
        self.data = np.load(data_file, mmap_mode='c').astype(np.float32)
        self.label = None
        if label_file is not None:
            self.label = np.load(label_file, allow_pickle=True).item()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_data = self.data[idx].transpose(2, 0, 1)
        sample = [image_data, {}]
        if self.label is not None:
            raw_label = self.label[idx]
            image_name = idx

            bbox = []
            for index in range(len(raw_label[0])):
                ele = raw_label[0][index]
                bbox.append(ele)
            easy = []
            for index in range(len(raw_label[1])):
                ele = raw_label[1][index]
                easy.append(ele)
            medium = []
            for index in range(len(raw_label[2])):
                ele = raw_label[2][index]
                medium.append(ele)
            hard = []
            for index in range(len(raw_label[3])):
                ele = raw_label[3][index]
                hard.append(ele)

            sample[1].update({
                'image_name': np.array(image_name),
                'bbox': np.array(bbox),
                'easy': np.array(easy),
                'medium': np.array(medium),
                'hard': np.array(hard),
            })
        return sample

    @staticmethod
    def collate_fn(batch):
        batch_label = {}
        batch_data = None
        for batch_idx in range(len(batch)):
            els_sample = batch[batch_idx][0]
            single_data = torch.unsqueeze(torch.tensor(els_sample), 0)
            batch_data = single_data if batch_idx == 0 else torch.cat(
                (batch_data, single_data), 0)

        image_list = []
        bbox_list = []
        easy_list, medium_list, hard_list = [], [], []
        for idx, sample in enumerate(batch):
            if not sample[1]:
                continue
            image_name = sample[1]['image_name']
            bbox = sample[1]['bbox']
            easy = sample[1]['easy']
            medium = sample[1]['medium']
            hard = sample[1]['hard']
            image_list.append(torch.tensor(image_name))
            bbox_list.append(torch.tensor([bbox]))
            easy_list.append(torch.tensor([easy]))
            medium_list.append(torch.tensor([medium]))
            hard_list.append(torch.tensor([hard]))
        batch_label.update({
            'image_name': image_list,
            'bbox': bbox_list,
            'easy': easy_list,
            'medium': medium_list,
            'hard': hard_list
        })
        return batch_data, batch_label
