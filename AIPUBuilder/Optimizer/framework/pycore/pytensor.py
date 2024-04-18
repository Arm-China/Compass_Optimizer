# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3

import torch
__all__ = [
    "PyTensor",
    "TensorShape",
    "get_tensor_default_property",
    "opt_use_cuda",
]

# import os
# cuda_device = os.environ['CUDA_VISIBLE_DEVICES']


def opt_use_cuda():
    import torch
    _USE_CUDA = False
    # if torch.cuda.is_available() and cuda_device != "":
    if torch.cuda.is_available():
        _USE_CUDA = True
    return _USE_CUDA


class TensorShape(tuple):
    def clone(self):
        import copy
        return copy.deepcopy(self)

    def dim(self):
        return len(self)

    def size(self):
        s = 1
        for ts in self:
            s *= ts
        return s


_tensor_default_property = {
    # quantization property
    "_scale": 1.0,
    "_zerop": 0,
    "qbits": None, "qmin": None,
    "qmax": None, "qinvariant": False,
    "dtype": None,
    # per-block
    "block_size": None,
    # per-channel
    "key_axis": 0,
    "key_axis_g": 1,
    "key_axis_bs": None,
    # source IR info
    "ir_shape": None, "ir_dtype": None,
    "ir_range": None,
    # producer node
    'pnode': None,
    # statistic property
    "extrema_min_key_axis": None,
    "extrema_max_key_axis": None,
    "running_min_key_axis": None,
    "running_max_key_axis": None,
    "running_mean_key_axis": None,
    "running_std_key_axis": None,
    "running_mad_key_axis": None,
    "running_histc_key_axis": None,
    "extrema_min": float("inf"),
    "extrema_max": float("-inf"),
    "running_min": 0.0,
    "running_max": 0.0,
    "running_mean": 0.0,
    "running_std": 0.0,
    "running_mad": 0.0,
    "running_histc": None,
    "_min": 0.0,
    "_max": 0.0,
    "min_key_axis": None,
    "max_key_axis": None,
    # records for dev
    "similarity": None,
    "mse": None,
    "debug_flag": 0,
    "need_deleted": False,
    "assigned_device": "cpu",
}


def get_tensor_default_property():
    return list(_tensor_default_property.keys())


class PyTensor:
    import torch
    import numpy as np
    from typing import Union
    from AIPUBuilder.Optimizer.framework.pycore.pytype import Dtype
    __slots__ = tuple(_tensor_default_property.keys()) + ('name', 'betensor', 'attrs')

    def __init__(self, name: str, shape_or_arr: Union[TensorShape, np.ndarray, torch.Tensor] = TensorShape(), dtype: Union[Dtype, None] = None):
        import torch
        import numpy as np
        from AIPUBuilder.Optimizer.framework.pycore.pytype import Dtype
        from AIPUBuilder.Optimizer.utils.dtype_utils import torch_type2dtype, str2dtype, torch_type2nptype
        th_dict = {
            None:                   None,
            Dtype.FP16:             torch.float16,
            Dtype.BFP16:            torch.bfloat16,
            Dtype.FP32:             torch.float32,
            Dtype.FP64:             torch.float32,  # fp32 is more efficient
            Dtype.BOOL:             torch.int32,  # in case of overflow in quant forward's computation
            Dtype.INT8:             torch.int32,  # in case of overflow in quant forward's computation
            Dtype.UINT8:            torch.int32,  # in case of overflow in quant forward's computation
            Dtype.INT16:            torch.int32,  # in case of overflow in quant forward's computation
            Dtype.UINT16:           torch.int32,  # no torch.uint16 and in case of overflow in quant forward's computation
            Dtype.INT32:            torch.int64,  # in case of overflow in quant forward's computation
            Dtype.UINT32:           torch.int64,  # no torch.uint32 and in case of overflow in quant forward's computation
            Dtype.INT64:            torch.int64,
            Dtype.UINT64:           torch.long,  # no torch.uint64
            Dtype.ALIGNED_INT4:     torch.int32,  # in case of overflow in quant forward's computation
            Dtype.ALIGNED_UINT4:    torch.int32,  # in case of overflow in quant forward's computation
            Dtype.ALIGNED_INT12:    torch.int32,  # in case of overflow in quant forward's computation
            Dtype.ALIGNED_UINT12:   torch.int32,  # in case of overflow in quant forward's computation
        }
        for k, v in _tensor_default_property.items():
            self.__setattr__(k, v)
        self.name = str(name)
        if isinstance(shape_or_arr, TensorShape):
            self.betensor = torch.zeros(shape_or_arr, dtype=th_dict[dtype])
            self.dtype = dtype
        elif isinstance(shape_or_arr, torch.Tensor):
            self.betensor = shape_or_arr.clone().detach()
            self.dtype = torch_type2dtype(self.betensor.dtype) if dtype is None else dtype
        elif shape_or_arr is None:
            self.betensor = torch.tensor(0, dtype=th_dict[dtype])
            self.dtype = dtype
        elif isinstance(shape_or_arr, np.ndarray):
            arr = shape_or_arr.astype(torch_type2nptype(th_dict[str2dtype(shape_or_arr.dtype.name)]))
            self.betensor = torch.tensor(arr, dtype=th_dict[dtype])
            self.dtype = str2dtype(shape_or_arr.dtype.name) if dtype is None else dtype
        else:
            self.betensor = torch.tensor(shape_or_arr, dtype=th_dict[dtype])
            self.dtype = torch_type2dtype(self.betensor.dtype) if dtype is None else dtype
        if opt_use_cuda():
            self.betensor = self.betensor.cuda()
            self.assigned_device = self.betensor.device
        self.fit_dtype()
        self.ir_shape = TensorShape(self.betensor.shape)
        self.block_size = None
        self.attrs = dict()

    def fit_dtype(self, dtype: Union[Dtype, None] = None):
        from AIPUBuilder.Optimizer.utils.dtype_utils import is_float, dtype2range, dtype2torch_type
        from AIPUBuilder.Optimizer.framework.pycore.pytype import Dtype
        from AIPUBuilder.Optimizer.logger import OPT_WARN, OPT_ERROR
        if dtype is not None:
            self.dtype = dtype

        if self.dtype is None:
            # no restricts
            pass
        elif is_float(self.dtype):
            if self.dtype in [Dtype.BFP16, Dtype.FP16, Dtype.FP32, Dtype.FP64]:
                self.betensor = self.betensor.to(dtype2torch_type(self.dtype))
            else:
                #self.betensor = to_fp24(self.betensor) if self.dtype == Dtype.FP24 else self.betensor
                OPT_ERROR(f'unsupported dtype "{self.dtype}" when calling fit_dtype function on tensor "{self.name}".')
        else:
            nan_mask = torch.isnan(self.betensor)
            if nan_mask.any():
                OPT_WARN(
                    f'tensor "{self.name}" contains NaN values and will be converted to zeros in fit_dtype() function.')
                self.betensor = torch.where(nan_mask, torch.zeros_like(self.betensor), self.betensor)
            qmin, qmax = dtype2range(self.dtype)
            self.betensor = torch.clamp(self.betensor, qmin, qmax).to(dtype2torch_type(self.dtype))
            if self.ir_dtype is not None and is_float(self.ir_dtype) and not self.qinvariant:
                # many torch api does not implement for none float dtypes
                self.betensor = self.betensor.float()

    def __repr__(self):
        import torch

        def _scale_zp(sz):
            _ret = None
            if isinstance(sz, (float, int)):
                _ret = sz
            elif isinstance(sz, torch.Tensor) and sz.numel() > 1:
                _ret = f"{sz[0]},...., {sz[-1]}"
            elif isinstance(sz, torch.Tensor):
                _ret = sz.item()
            return _ret
        scale = _scale_zp(self.scale)
        zerop = _scale_zp(self.zerop)
        return (f"tensor.name={self.name}, tensor.ir_shape={self.ir_shape}, tensor.dtype={self.dtype}, "
                f"tensor.qinfo[scale, zerop] = [{scale}, {zerop}]")

    def clone(self, name=None):
        import copy
        import torch
        if name is None:
            name = self.name + '_clone' if not self.name.endswith("_clone") else self.name
        t = self.__class__(name, self.betensor)
        for k in _tensor_default_property.keys():
            v = self.__getattribute__(k)
            if isinstance(v, torch.Tensor):
                t.__setattr__(k, v.clone().detach())
            elif k in ['pnode']:
                t.__setattr__(k, None)
            else:
                t.__setattr__(k, copy.deepcopy(v))
        for k, v in self.attrs.items():
            t.attrs[k] = copy.deepcopy(v)
        return t

    def statistic(self,
                  running_statistic_momentum,
                  key_axis=None,  # None means not statistic per-channel info
                  key_axis_g=1,
                  histc_bins=None,  # None means not statistic histogram
                  statistic_std_mean=True,
                  # How to deal with infinite or equivalent very large/small values
                  trim_infinity=((float('-inf'), float('inf')), ''),
                  reset=False):
        import torch
        from AIPUBuilder.Optimizer.utils import OPT_INT_MAX, OPT_INT_MIN, OPT_EPSILON
        from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
        tdevice = self.device
        fbetensor = self.betensor.float()
        OPT_INT_MIN_t = torch_tensor(OPT_INT_MIN, device=tdevice)
        OPT_INT_MAX_t = torch_tensor(OPT_INT_MAX, device=tdevice)

        if trim_infinity[-1] == 'clip':
            fbetensor = torch.clamp(fbetensor.clone().detach(), trim_infinity[0][0], trim_infinity[0][1])
        elif trim_infinity[-1] == 'second':
            trim_min, trim_max = min(trim_infinity[0][0], 0.0), max(trim_infinity[0][1], 0.0)
            sfbetensor = torch.where(fbetensor <= trim_min, torch.zeros_like(fbetensor), fbetensor)
            sfbetensor = torch.where(sfbetensor >= trim_max, torch.zeros_like(sfbetensor), sfbetensor)
            second_min = sfbetensor.min()
            second_max = sfbetensor.max()
            fbetensor = torch.clamp(fbetensor.clone().detach(), second_min, second_max)
        else:
            pass

        bmin = fbetensor.min()
        bmax = fbetensor.max()
        bmin = max(bmin, OPT_INT_MIN_t)
        bmax = min(bmax, OPT_INT_MAX_t)
        other_dims = [i for i in range(fbetensor.dim())]
        channels = 0

        if key_axis is not None and fbetensor.dim() > 0:
            other_dims.remove(other_dims[key_axis])
            channels = fbetensor.shape[key_axis]
        momentum = running_statistic_momentum
        if reset:
            momentum = 0.0
            if key_axis is not None:
                self.extrema_min_key_axis = torch.full([channels], float("inf"), device=tdevice)
                self.extrema_max_key_axis = torch.full([channels], float("-inf"), device=tdevice)
                self.running_min_key_axis = torch.zeros([channels], device=tdevice)
                self.running_max_key_axis = torch.zeros([channels], device=tdevice)

                if statistic_std_mean:
                    self.running_mean_key_axis = torch.zeros([channels], device=tdevice)
                    self.running_std_key_axis = torch.zeros([channels], device=tdevice)
                    self.running_mad_key_axis = torch.zeros([channels], device=tdevice)
                if histc_bins != None:
                    self.running_histc_key_axis = torch.zeros([channels, histc_bins], device=tdevice)
            self.extrema_min = torch_tensor(float("inf"), device=tdevice)
            self.extrema_max = torch_tensor(float("-inf"), device=tdevice)
            self.running_min = torch.tensor(0.0, device=tdevice)
            self.running_max = torch.tensor(0.0, device=tdevice)
            if statistic_std_mean:
                self.running_mean = 0.0
                self.running_std = 0.0
                self.running_mad = 0.0
            if histc_bins is not None:
                self.running_histc = torch.zeros([histc_bins], device=tdevice)
            # if key_axis != None :
            #     self.clip_min_key_axis = None #torch.tensor([0.0 for i in range(channels)], device=tdevice)
            #     self.clip_max_key_axis = None #torch.tensor([0.0 for i in range(channels)], device=tdevice)
            # self.clip_min = None
            # self.clip_max = None

        if key_axis is not None:
            perm = [key_axis] + [axis for axis in range(fbetensor.dim()) if axis != key_axis]
            g, gn = channels, 1
            if key_axis_g > 1:
                gn = key_axis_g
                # TODO: if no divide exactly, should be padded the extrema value in min or max value
                g = int(self.betensor.shape[key_axis] / gn)
            torch_int_min = torch.tensor(OPT_INT_MIN, device=tdevice)
            torch_int_max = torch.tensor(OPT_INT_MAX, device=tdevice)
            _min_key_axis = fbetensor.permute(perm).reshape([channels, -1]).min(dim=-1).values
            _min_key_axis = _min_key_axis.reshape([gn, g]).min(dim=0).values
            _min_key_axis = torch.repeat_interleave(_min_key_axis, gn)
            running_min_key_axis = torch.maximum(_min_key_axis, torch_int_min)

            self.extrema_min_key_axis = torch.min(self.extrema_min_key_axis, running_min_key_axis)
            self.running_min_key_axis = momentum * self.running_min_key_axis + (1.0-momentum) * running_min_key_axis

            _max_key_axis = fbetensor.permute(perm).reshape([channels, -1]).max(dim=-1).values
            _max_key_axis = _max_key_axis.reshape([gn, g]).max(dim=0).values
            _max_key_axis = torch.repeat_interleave(_max_key_axis, gn)
            running_max_key_axis = torch.minimum(_max_key_axis, torch_int_max)

            self.extrema_max_key_axis = torch.max(self.extrema_max_key_axis, running_max_key_axis)
            self.running_max_key_axis = momentum * self.running_max_key_axis + (1.0-momentum) * running_max_key_axis
            if statistic_std_mean:
                running_std_key_axis, running_mean_key_axis = torch.std_mean(fbetensor, dim=other_dims)
                running_mad_key_axis = torch.mean(
                    torch.abs(fbetensor - torch.mean(fbetensor, dim=other_dims, keepdim=True)), dim=other_dims)
                running_std_key_axis = torch.clamp(running_std_key_axis, OPT_INT_MIN, OPT_INT_MAX)
                running_mean_key_axis = torch.clamp(running_mean_key_axis, OPT_INT_MIN, OPT_INT_MAX)
                running_mad_key_axis = torch.clamp(running_mad_key_axis, OPT_INT_MIN, OPT_INT_MAX)
                running_std_key_axis[torch.isnan(running_std_key_axis)] = 1.0
                running_mean_key_axis[torch.isnan(running_mean_key_axis)] = 0.0
                running_mad_key_axis[torch.isnan(running_mad_key_axis)] = 0.0
                self.running_mean_key_axis = momentum * self.running_mean_key_axis + \
                    (1.0-momentum) * running_mean_key_axis
                self.running_std_key_axis = momentum * self.running_std_key_axis + (1.0-momentum) * running_std_key_axis
                self.running_mad_key_axis = momentum * self.running_mad_key_axis + (1.0-momentum) * running_mad_key_axis
            if histc_bins != None:
                running_histc_key_axis = torch.zeros_like(self.running_histc_key_axis)
                for i in range(channels):
                    kmin = max(min(running_min_key_axis[i], OPT_INT_MAX), OPT_INT_MIN)
                    kmax = min(max(running_max_key_axis[i], OPT_INT_MIN), OPT_INT_MAX)
                    if kmin != kmin:
                        # nan
                        kmin = 0
                    if kmax != kmax:
                        # nan
                        kmax = 0
                    if kmax <= kmin:
                        kmax = kmin + abs(kmin / 2.) + 1.
                    running_histc_key_axis[i] = fbetensor.select(
                        key_axis, i).float().histc(bins=histc_bins, min=kmin, max=kmax)
                self.running_histc_key_axis = momentum * self.running_histc_key_axis + \
                    (1.0-momentum) * running_histc_key_axis
        self.extrema_min = min(self.extrema_min.to(torch.float32), bmin)
        self.extrema_max = max(self.extrema_max.to(torch.float32), bmax)
        self.running_min = momentum * self.running_min + (1.0 - momentum) * bmin
        self.running_max = momentum * self.running_max + (1.0 - momentum) * bmax
        if statistic_std_mean:
            running_std, running_mean = torch.std_mean(fbetensor)
            running_mad = torch.mean(torch.abs(fbetensor - running_mean))
            running_std = torch.clamp(running_std, OPT_INT_MIN_t, OPT_INT_MAX_t)
            running_mean = torch.clamp(running_mean, OPT_INT_MIN_t, OPT_INT_MAX_t)
            running_mad = torch.clamp(running_mad, OPT_INT_MIN_t, OPT_INT_MAX_t)
            running_std[torch.isnan(running_std)] = 1.0
            running_mean[torch.isnan(running_mean)] = 0.0
            running_mad[torch.isnan(running_mad)] = 0.0
            self.running_mean = momentum * self.running_mean + (1.0 - momentum) * running_mean
            self.running_std = momentum * self.running_std + (1.0 - momentum) * running_std
            self.running_mad = momentum * self.running_mad + (1.0 - momentum) * running_mad
        if histc_bins is not None:
            kmin = max(min(bmin, OPT_INT_MAX_t), OPT_INT_MIN_t)
            kmax = min(max(bmax, OPT_INT_MIN_t), OPT_INT_MAX_t)
            if kmin != kmin:
                # nan
                kmin = 0
            if kmax != kmax:
                # nan
                kmax = 0
            if kmax <= kmin:
                kmax = kmin + abs(kmin / 2.) + 1.
            self.running_histc = momentum * self.running_histc + \
                (1.0 - momentum) * fbetensor.histc(bins=histc_bins, min=kmin, max=kmax)

    def key_axis_broadcast_shape(self):
        """
        generate one shape, like [1, 1, -1, 1] when tensor.key_axis = 2, this shape can use for
        reshape the scale or zerop tensor.
        """
        # key_axis_shape = [] # per-tensor, scale=torch.tensor(2.0)
        key_axis_shape = [-1]  # per-tensor, scale=torch.tensor([2.0])
        if self.key_axis is not None and len(self.ir_shape) > 0:  # pylint: disable=no-member
            key_axis_shape = [1] * len(self.ir_shape)
            key_axis_shape[self.key_axis] = -1  # pylint: disable=no-member
        return key_axis_shape

    def set_qinvariant(self, qinvariant=None):
        """
        when meets scale=1.0, zp=0, can use this interface to set tensor.qinvariant=True
        """
        import torch
        if qinvariant is not None:
            self.qinvariant = qinvariant
            return
        one_t = torch.ones_like(self.scale)
        zero_t = torch.zeros_like(self.zerop)
        if (self.scale == one_t).all() and (self.zerop == zero_t).all():
            self.qinvariant = True

    def is_qinvariant(self):
        """
        return tensor whether qinvariant
        """
        import torch
        ret = self.qinvariant
        one_t = torch.ones_like(self.scale)
        zero_t = torch.zeros_like(self.zerop)
        ret = ret and (self.scale == one_t).all() and (self.zerop == zero_t).all()
        return ret

    def is_perchannel_scales(self):
        import torch
        from AIPUBuilder.Optimizer.utils import is_torch_tensor_with_multi_data
        return True if is_torch_tensor_with_multi_data(self.scale) else False

    def is_perchannel_zerops(self):
        import torch
        from AIPUBuilder.Optimizer.utils import is_torch_tensor_with_multi_data
        return True if is_torch_tensor_with_multi_data(self.zerop) else False

    def is_perchannel_quantization(self):
        return True if self.key_axis is not None else False  # pylint: disable=no-member

    @property
    def scale(self):
        from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
        if self._scale is not None:
            self._scale = torch_tensor(self._scale, device=self.device)
            self._scale = self._scale.reshape([-1]).float()
        return self._scale  # pylint: disable=no-member

    @scale.setter
    def scale(self, scale):
        import torch
        from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
        if isinstance(scale, torch.Tensor):
            self._scale = scale.reshape([-1]).float().to(self.device)
        elif scale is None:
            self._scale = None
        else:
            self.scale = torch_tensor(scale, device=self.device)

    @property
    def zerop(self):
        from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
        if self._zerop is not None:
            self._zerop = torch_tensor(self._zerop, device=self.device)
            self._zerop = self._zerop.reshape([-1])
        return self._zerop  # pylint: disable=no-member

    @zerop.setter
    def zerop(self, zerop):
        import torch
        from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
        if isinstance(zerop, torch.Tensor):
            self._zerop = zerop.reshape([-1]).int().to(self.device)
        elif zerop is None:
            self._zerop = zerop
        else:
            self.zerop = torch_tensor(zerop, device=self.device)

    @property
    def broadcast_scale(self):
        if self.block_size is not None and len(self.ir_shape) > 0:
            bshape = list(self.ir_shape)
            bshape[-1] = int(self.ir_shape[-1] / self.block_size)
            return self.scale.flatten().reshape(bshape).repeat_interleave(self.block_size, dim=-1)
        else:
            return self.scale.reshape(self.key_axis_broadcast_shape()) if self.scale is not None else self.scale  # pylint: disable=no-member

    @broadcast_scale.setter
    def broadcast_scale(self, broadcast_scale):
        self.scale = broadcast_scale

    @property
    def broadcast_zerop(self):
        if self.block_size is not None and len(self.ir_shape) > 0:
            bshape = list(self.ir_shape)
            bshape[-1] = int(self.ir_shape[-1] / self.block_size)
            return self.zerop.flatten().reshape(bshape).repeat_interleave(self.block_size, dim=-1)
        else:
            return self.zerop.reshape(self.key_axis_broadcast_shape()) if self.zerop is not None else self.zerop  # pylint: disable=no-member

    @broadcast_zerop.setter
    def broadcast_zerop(self, broadcast_zerop):  # same as self.zerop = zp
        self.zerop = broadcast_zerop

    @property
    def max(self):
        from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
        if self._max is not None:
            self._max = torch_tensor(self._max, device=self.device)
            self._max = self._max.reshape([-1])
        return self._max  # pylint: disable=no-member

    @max.setter
    def max(self, max_v):
        import torch
        from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
        if isinstance(max_v, torch.Tensor):
            self._max = max_v.reshape([-1])
        elif max is None:
            self._max = None
        else:
            self.max = torch_tensor(max_v, device=self.device)

    @property
    def min(self):
        from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
        if self._min is not None:
            self._min = torch_tensor(self._min, device=self.device)
            self._min = self._min.reshape([-1])
        return self._min  # pylint: disable=no-member

    @min.setter
    def min(self, min_v):
        import torch
        from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
        if isinstance(min_v, torch.Tensor):
            self._min = min_v.reshape([-1])
        elif min is None:
            self._min = None
        else:
            self.min = torch_tensor(min_v, device=self.device)

    @property
    def device(self):
        return self.betensor.device


PyTensor.shape = property(lambda self: self.betensor.shape, None)
