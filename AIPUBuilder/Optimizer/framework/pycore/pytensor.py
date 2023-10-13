# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3

__all__ = [
    "PyTensor",
    "TensorShape",
    "get_tensor_default_property",
    "opt_use_cuda",
]


def opt_use_cuda():
    import torch
    _USE_CUDA = False
    if torch.cuda.is_available():
        _USE_CUDA = True
    return _USE_CUDA


class TensorShape(tuple):
    def clone(self):
        import copy
        return copy.deepcopy(self)


_tensor_default_property = {
    # quantization property
    "scale": 1., "zerop": 0, "qbits": None, "qmin": None,
    "qmax": None, "qinvariant": False,
    "dtype": None,
    # per-channel
    "key_axis": 0, "key_axis_g": None, "key_axis_bs": None,
    # source IR info
    "ir_shape": None, "ir_dtype": None,
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
    "min": 0.0,
    "max": 0.0,
    "min_key_axis": None,
    "max_key_axis": None,
    # records for dev
    "similarity": None,
    "debug_flag": 0,
    "need_deleted": False,
}


def get_tensor_default_property():
    return list(_tensor_default_property.keys())


class PyTensor:
    import torch
    import numpy as np
    from typing import Union
    from AIPUBuilder.Optimizer.framework.pycore.pytype import Dtype
    __slots__ = tuple(_tensor_default_property.keys()) + ('name', 'betensor')

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
        self.ir_shape = TensorShape(self.betensor.shape)

    def clone(self, name=None):
        import copy
        import torch
        if name is None:
            name = self.name + '_clone'
        t = self.__class__(name, self.betensor)
        for k in _tensor_default_property.keys():
            v = self.__getattribute__(k)
            if isinstance(v, torch.Tensor):
                t.__setattr__(k, v.clone().detach())
            elif k in ['pnode']:
                t.__setattr__(k, None)
            else:
                t.__setattr__(k, copy.deepcopy(v))
        return t

    def statistic(self,
                  running_statistic_momentum,
                  key_axis=None,  # None means not statistic per-channel info
                  histc_bins=None,  # None means not statistic histogram
                  statistic_std_mean=True,
                  # How to deal with infinite or equivalent very large/small values
                  trim_infinity=((float('-inf'), float('inf')), ''),
                  reset=False):
        import torch
        from AIPUBuilder.Optimizer.utils import OPT_INT_MAX, OPT_INT_MIN
        tdevice = self.betensor.device
        fbetensor = self.betensor.float()

        if trim_infinity[-1] == 'clip':
            fbetensor = torch.clamp(fbetensor.clone().detach(), trim_infinity[0][0], trim_infinity[0][1])
        elif trim_infinity[-1] == 'second':
            trim_min, trim_max = min(trim_infinity[0][0], 0.0), max(trim_infinity[0][1], 0.0)
            sfbetensor = torch.where(fbetensor <= trim_min, torch.zeros_like(fbetensor), fbetensor)
            sfbetensor = torch.where(sfbetensor >= trim_max, torch.zeros_like(sfbetensor), sfbetensor)
            second_min = sfbetensor.min().item()
            second_max = sfbetensor.max().item()
            fbetensor = torch.clamp(fbetensor.clone().detach(), second_min, second_max)
        else:
            pass

        bmin = fbetensor.min().item()
        bmax = fbetensor.max().item()
        bmin = max(bmin, OPT_INT_MIN)
        bmax = min(bmax, OPT_INT_MAX)
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
            self.extrema_min = float("inf")
            self.extrema_max = float("-inf")
            self.running_min = 0.0
            self.running_max = 0.0
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
            torch_int_min = torch.tensor(OPT_INT_MIN, device=tdevice)
            torch_int_max = torch.tensor(OPT_INT_MAX, device=tdevice)
            running_min_key_axis = torch.maximum(fbetensor.permute(perm).reshape([channels, -1]).min(dim=-1).values,
                                                 torch_int_min)
            self.extrema_min_key_axis = torch.min(self.extrema_min_key_axis, running_min_key_axis)
            self.running_min_key_axis = momentum * self.running_min_key_axis + (1.0-momentum) * running_min_key_axis
            running_max_key_axis = torch.minimum(fbetensor.permute(perm).reshape([channels, -1]).max(dim=-1).values,
                                                 torch_int_max)
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
        self.extrema_min = min(float(self.extrema_min), bmin)
        self.extrema_max = max(float(self.extrema_max), bmax)
        self.running_min = momentum * self.running_min + (1.0-momentum) * bmin
        self.running_max = momentum * self.running_max + (1.0-momentum) * bmax
        if statistic_std_mean:
            running_std, running_mean = torch.std_mean(fbetensor)
            running_mad = torch.mean(torch.abs(fbetensor - running_mean))
            running_std = torch.clamp(running_std, OPT_INT_MIN, OPT_INT_MAX)
            running_mean = torch.clamp(running_mean, OPT_INT_MIN, OPT_INT_MAX)
            running_mad = torch.clamp(running_mad, OPT_INT_MIN, OPT_INT_MAX)
            running_std[torch.isnan(running_std)] = 1.0
            running_mean[torch.isnan(running_mean)] = 0.0
            running_mad[torch.isnan(running_mad)] = 0.0
            self.running_mean = momentum * self.running_mean + (1.0-momentum) * running_mean.item()
            self.running_std = momentum * self.running_std + (1.0-momentum) * running_std.item()
            self.running_mad = momentum * self.running_mad + (1.0-momentum) * running_mad.item()
        if histc_bins != None:
            kmin = max(min(bmin, OPT_INT_MAX), OPT_INT_MIN)
            kmax = min(max(bmax, OPT_INT_MIN), OPT_INT_MAX)
            if kmin != kmin:
                # nan
                kmin = 0
            if kmax != kmax:
                # nan
                kmax = 0
            if kmax <= kmin:
                kmax = kmin + abs(kmin / 2.) + 1.
            self.running_histc = momentum * self.running_histc + \
                (1.0-momentum) * fbetensor.histc(bins=histc_bins,
                                                 min=kmin, max=kmax)  # running_histc_key_axis.sum(dim=0)

    def __repr__(self):
        import torch

        def _format(sz):
            _ret = None
            if isinstance(sz, (float, int)):
                _ret = sz
            elif isinstance(sz, torch.Tensor) and sz.numel() > 1:
                _ret = f"{sz[0]},...., {sz[-1]}"
            elif isinstance(sz, torch.Tensor):
                _ret = sz.item()
            return _ret
        scale = _format(self.scale)
        zerop = _format(self.zerop)
        return (f"tensor.name={self.name}, tensor.ir_shape={self.ir_shape}, "
                f"tensor.scale={scale}, tensor.zerop={zerop}")


PyTensor.shape = property(lambda self: self.betensor.shape, None)
