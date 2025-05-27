# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from torch.utils.data import Dataset
import numpy as np


@register_plugin(PluginType.Dataset, '1.0')
class MpiiDataset(Dataset):
    """
    This MpiiDataset plugin is mainly used for stacked_hourglass_tensorflow model in Optimizer.
    MPII Human Pose dataset is a state of the art benchmark for evaluation of articulated human pose estimation.
    The dataset includes around 25K images containing over 40K people with annotated body joints.
    http://human-pose.mpi-inf.mpg.de/
    """

    def __init__(self, data_file=None, label_file=None):
        '''
        :param data_file: ndarray in npy file
        :param label_file: a dict format in npy file and the keys of dict include
        ['__header__', '__version__', '__globals__', 'jnt_missing', 'pos_gt_src', 'headboxes_src', 'center', 'scale'].
        '''
        self.data = np.load(data_file, allow_pickle=True)
        self.label = None
        if label_file is not None:
            self.label = np.load(label_file, allow_pickle=True).tolist()
            self.keys = [k for k in self.label if '__' not in k]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = [self.data[idx], {}]
        if self.label is not None:
            pick_label = {}
            for k in self.keys:
                if k in ['center', 'scale']:
                    pick_label[k] = self.label[k][idx]
                else:
                    pick_label[k] = self.label[k][..., idx]
            sample[1] = pick_label
        return sample
