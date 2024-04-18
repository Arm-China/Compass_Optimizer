# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate
import numpy as np
import torch

TORCH_VERSION = torch.__version__


def digit_version(version_str):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions.

    Args:
        version_str (str): The version string.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return tuple(digit_version)


@register_plugin(PluginType.Dataset, '1.0')
class BevFormerDataset(Dataset):

    def __init__(self, data_file=None, label_file=None):
        self.prev_bev_flag = None
        self.imgs = []
        self.prev_bev = []
        self.can_bus = []
        self.lidar2img = []
        self.scene_token = []
        self.prev_bev_shape = [2500, 1, 256]
        self.can_bus_shape = [18]
        self.lidar2img_shape = [6, 4, 4]
        self.img_shape = [1, 6, 3, 480, 800]
        self.point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.voxel_size = [0.2, 0.2, 8]
        self.bev_h = 50
        self.bev_w = 50
        self.num_points_in_pillar = 4
        self.real_w = self.point_cloud_range[3] - self.point_cloud_range[0]
        self.real_h = self.point_cloud_range[4] - self.point_cloud_range[1]
        self.grid_length = [self.real_h / self.bev_h, self.real_w / self.bev_w]
        self.prev_tmp_angle = None
        self.prev_tmp_pos = None
        self.use_shift = True
        self.rotation_prev_bev = True
        self.rotate_center = [100, 100]
        self.consumer = None

        self.data = np.load(data_file, allow_pickle=True).tolist()
        self.label = None
        if label_file is not None:
            self.label = np.load(label_file, allow_pickle=True)

        self.data_process()

    def data_process(self):
        for pid, data in self.data.items():
            img = data['img']
            img_metas = data['img_metas'][0]

            scene_token = img_metas['scene_token']
            can_bus = img_metas['can_bus']
            lidar2img = np.stack(img_metas['lidar2img'], axis=0)  # shape (6, 4, 4)

            prev_bev = np.zeros([1])

            self.imgs.append(img)
            self.can_bus.append(can_bus)
            self.lidar2img.append(lidar2img)
            self.prev_bev.append(prev_bev)
            self.scene_token.append(scene_token)

        self.imgs = np.concatenate(self.imgs, axis=0)
        self.can_bus = np.stack(self.can_bus, axis=0)
        self.lidar2img = np.stack(self.lidar2img, axis=0)
        self.prev_bev = np.stack(self.prev_bev, axis=0)

    def align_prev_bev(self, can_bus, prev_bev, batch=1, use_shift=True, new_prev=True):
        delta_x = can_bus[0]
        delta_y = can_bus[1]

        ego_angle = can_bus[-2] / np.pi * 180
        grid_length_y = self.grid_length[0]
        grid_length_x = self.grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / self.bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / self.bev_w
        shift_y = shift_y * use_shift
        shift_x = shift_x * use_shift

        shift = torch.tensor([shift_x, shift_y]).reshape([-1, 1]).permute(1, 0)

        if self.rotation_prev_bev and not new_prev:
            if batch > 1:
                OPT_ERROR(f"now only support batch==1")
            for i in range(batch):
                rotation_angle = can_bus[-1]
                tmp_prev_bev = torch_tensor(prev_bev[:, i]).reshape(self.bev_h, self.bev_w, -1).permute(2, 0, 1)
                tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(self.bev_h * self.bev_w, 1, -1)
                prev_bev = tmp_prev_bev

        return prev_bev, shift

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def generate_lidar2img(self, lidar2img):
        ref_3d = BevFormerDataset.get_reference_points(
            self.bev_h, self.bev_w, self.point_cloud_range[5] - self.point_cloud_range[2], self.num_points_in_pillar)
        ref_point_cam, bev_mask = self.point_sampling(lidar2img, ref_3d, self.point_cloud_range)
        return ref_point_cam, bev_mask

    def point_sampling(self, lidar2img, reference_points, pc_range, img_metas=None):

        # lidar2img = []
        # for img_meta in img_metas:
        #     lidar2img.append(img_meta['lidar2img'])
        # lidar2img = np.asarray(lidar2img)
        ld2img = lidar2img.reshape([1, *lidar2img.shape])
        ld2img = reference_points.new_tensor(ld2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = ld2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        ld2img = ld2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(ld2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        # eps = 1e-5
        # # eps = 1e-2
        eps = 1e-1
        # eps = 1

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        # reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
        reference_points_cam[..., 0] /= self.img_shape[-1]
        reference_points_cam[..., 1] /= self.img_shape[-2]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        # reference_points_cam_p = reference_points_cam[bev_mask]
        # print(f"cam_range = {reference_points_cam_p.min() * 2 -1}, {reference_points_cam_p.max()*2-1}, before mask {reference_points_cam.min()*2-1}, {reference_points_cam.max()*2-1}")

        # lift bn and eliminate the split_concat_tile
        reference_ = torch.split(reference_points_cam, [1] * reference_points_cam.shape[0], dim=0)
        reference_ = torch.cat(reference_, dim=1)
        ref_trans = reference_.permute(1, 0, 2, 3, 4).reshape(6, 2500, 1, 1, 1, 4, 2)
        reference_points_cam = ref_trans.repeat([1, 1, 8, 1, 2, 1, 1]) * 2 - 1
        clip_v = reference_points_cam.new_tensor(25.)
        reference_points_cam = torch.clamp(reference_points_cam, -clip_v, clip_v)

        return reference_points_cam, bev_mask.int()

    def __len__(self):
        return self.imgs.shape[0]

    def reset(self):
        self.prev_bev_flag = None
        # self.consumer = None

    def __getitem__(self, idx):
        # input_tensors = [lidar2img_0, img_0, can_bus_0, pre_bev_0]
        if idx == 0:
            self.reset()
        # sample = [[self.lidar2img[idx], self.imgs[idx], None, self.prev_bev[idx]], []]
        # sample = [[self.lidar2img[idx], self.imgs[idx], None, 0, self.prev_bev[idx]], []]
        sample = [[self.imgs[idx], None, 0, None, None, self.prev_bev[idx]], []]
        can_bus = self.can_bus[idx]
        self.prev_tmp_pos = can_bus[:3]
        self.prev_tmp_angle = can_bus[-1]
        if self.scene_token[idx] != self.prev_bev_flag:
            # OPT_ERROR(f"{idx} self.pre_bev_flag == {self.prev_bev_flag}")
            prev_bev = np.zeros(self.prev_bev_shape)
            # the newest can_bus data has preprocess the can_bus issue, because we save the metas data before img_feat()
            # can_bus[-1] = 0
            # can_bus[:3] = 0
        else:
            if self.consumer is not None and isinstance(self.consumer, PyGraph):
                # OPT_ERROR(f"{idx} BEVFormer Dataset has consumer {self.consumer}")
                prev_bev = self.consumer.output_tensors[0].betensor.detach()
            else:
                prev_bev = np.zeros(self.prev_bev_shape)
            # the newest can_bus data has preprocess the can_bus issue, because we save the metas data before img_feat()
            # can_bus[:3] -= self.prev_tmp_pos
            # can_bus[-1] -= self.prev_tmp_angle

        prev_bev, shift = self.align_prev_bev(
            can_bus, prev_bev, new_prev=True if self.scene_token[idx] != self.prev_bev_flag else False)
        self.prev_bev_flag = self.scene_token[idx]

        ref_point_cam, bev_mask = self.generate_lidar2img(self.lidar2img[idx])
        # sample[0][3] = prev_bev

        # sample[0][3] = shift
        # sample[0][4] = prev_bev
        #
        # sample[0][2] = can_bus

        sample[0][1] = can_bus
        sample[0][2] = shift
        sample[0][3] = ref_point_cam
        sample[0][4] = bev_mask
        sample[0][5] = prev_bev
        return sample


if __name__ == '__main__':
    dataset = BevFormerDataset('/project/ai/zhouyi_compass/samhan01/work/models/bevformer/dataset_new_2.npy')
    print(dataset)
    d0 = dataset[0]
