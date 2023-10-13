# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

import torch
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import OPT_FATAL

OPT_EPSILON = torch.finfo(torch.float32).eps
OPT_INT_MIN = -2 ** 31
OPT_INT_MAX = 2 ** 31 - 1


def construct_torch_tensor(var, device=None):
    if isinstance(var, torch.Tensor):
        return var.to(device=device)
    else:
        return torch.tensor(var, device=device)


def nhwc2nchw(x):
    return x.permute(0, 3, 1, 2)


def nchw2nhwc(x):
    return x.permute(0, 2, 3, 1)


def is_signed(dt):
    dt_dict = {
        Dtype.FP16:     True,
        Dtype.FP32:     True,
        Dtype.FP64:     True,
        Dtype.INT8:     True,
        Dtype.UINT8:    False,
        Dtype.INT16:    True,
        Dtype.UINT16:   False,
        Dtype.INT32:    True,
        Dtype.UINT32:   False,
        Dtype.INT64:    True,
        Dtype.UINT64:   False,
        Dtype.ALIGNED_INT4: True,
        Dtype.ALIGNED_UINT4: False,
        Dtype.ALIGNED_INT12: True,
        Dtype.ALIGNED_UINT12: False,
    }
    th_dict = {
        torch.bfloat16:    True,
        torch.float16:     True,
        torch.float32:     True,
        torch.float64:     True,
        torch.int8:        True,
        torch.uint8:       False,
        torch.int16:       True,
        # torch.uint16:      False,
        torch.int32:       True,
        # torch.uint32:      False,
        torch.int64:       True,
        # torch.uint64:      False,
        torch.long:        True,
        torch.int:         True,
    }
    if dt in dt_dict:
        return dt_dict[dt]
    else:
        return th_dict[dt]


def is_float(dt):
    dt_dict = {
        Dtype.BFP16:    True,
        Dtype.FP16:     True,
        Dtype.FP32:     True,
        Dtype.FP64:     True,
        Dtype.INT8:     False,
        Dtype.UINT8:    False,
        Dtype.INT16:    False,
        Dtype.UINT16:   False,
        Dtype.INT32:    False,
        Dtype.UINT32:   False,
        Dtype.INT64:    False,
        Dtype.UINT64:   False,
        Dtype.ALIGNED_INT4: False,
        Dtype.ALIGNED_UINT4: False,
        Dtype.ALIGNED_INT12: False,
        Dtype.ALIGNED_UINT12: False,
    }
    th_dict = {
        torch.bfloat16:    True,
        torch.float16:     True,
        torch.float32:     True,
        torch.float64:     True,
        torch.int8:        False,
        torch.uint8:       False,
        torch.int16:       False,
        # torch.uint16:      False,
        torch.int32:       False,
        # torch.uint32:      False,
        torch.int64:       False,
        # torch.uint64:      False,
        torch.long:        False,
        torch.int:         False,
    }
    if dt in dt_dict:
        return dt_dict[dt]
    else:
        return th_dict[dt]


def bits2dtype(bits, is_signed, use_float=False):
    if bits <= 4:
        if use_float:
            return Dtype.BFP16
        else:
            return Dtype.ALIGNED_INT4 if is_signed else Dtype.ALIGNED_UINT4
    elif bits <= 8:
        if use_float:
            return Dtype.BFP16
        else:
            return Dtype.INT8 if is_signed else Dtype.UINT8
    elif bits <= 12:
        if use_float:
            return Dtype.BFP16
        else:
            return Dtype.ALIGNED_INT12 if is_signed else Dtype.ALIGNED_UINT12
    elif bits <= 16:
        if use_float:
            return Dtype.BFP16
        else:
            return Dtype.INT16 if is_signed else Dtype.UINT16
    elif bits <= 32:
        if use_float:
            return Dtype.FP32
        else:
            return Dtype.INT32 if is_signed else Dtype.UINT32
    else:
        if use_float:
            return Dtype.FP64
        else:
            return Dtype.INT64 if is_signed else Dtype.UINT64


def str2dtype(name):
    dname = name.strip().lower()
    if dname in ['float16', 'fp16', 'dtype.float16', 'dtype.fp16', ]:
        return Dtype.FP16
    elif dname in ['bfloat16', 'bf16', 'bfp16', 'dtype.bfloat16', 'dtype.bf16', 'dtype.bfp16', ]:
        return Dtype.BFP16
    elif dname in ['float', 'float32', 'fp32', 'dtype.float', 'dtype.float32', 'dtype.fp32', ]:
        return Dtype.FP32
    elif dname in ['double', 'float64', 'fp64', 'dtype.double', 'dtype.float64', 'dtype.fp64', ]:
        return Dtype.FP64
    elif dname in ['bool', 'boolean', 'dtype.bool', 'dtype.boolean', ]:
        return Dtype.BOOL
    elif dname in ['int8', 'dtype.int8', ]:
        return Dtype.INT8
    elif dname in ['uint8', 'dtype.uint8', ]:
        return Dtype.UINT8
    elif dname in ['int16', 'dtype.int16', ]:
        return Dtype.INT16
    elif dname in ['uint16', 'dtype.uint16', ]:
        return Dtype.UINT16
    elif dname in ['int32', 'dtype.int32', ]:
        return Dtype.INT32
    elif dname in ['uint32', 'dtype.uint32', ]:
        return Dtype.UINT32
    elif dname in ['int64', 'dtype.int64', ]:
        return Dtype.INT64
    elif dname in ['uint64', 'dtype.uint64', ]:
        return Dtype.UINT64
    elif dname in ['int4', 'aligned_int4', 'dtype.int4', 'dtype.aligned_int4', ]:
        return Dtype.ALIGNED_INT4
    elif dname in ['uint4', 'aligned_uint4', 'dtype.uint4', 'dtype.aligned_uint4', ]:
        return Dtype.ALIGNED_UINT4
    elif dname in ['int12', 'aligned_int12', 'dtype.int12', 'dtype.aligned_int12', ]:
        return Dtype.ALIGNED_INT12
    elif dname in ['uint12', 'aligned_uint12', 'dtype.uint12', 'dtype.aligned_uint12', ]:
        return Dtype.ALIGNED_UINT12
    else:
        OPT_FATAL('unsupported dtype: %s' % name)
        return None


def dtype2str(dt):
    dt_dict = {
        Dtype.BFP16:    'bfloat16',
        Dtype.FP16:     'float16',
        Dtype.FP32:     'float32',
        Dtype.FP64:     'float64',
        Dtype.BOOL:     'bool',
        Dtype.INT8:     'int8',
        Dtype.UINT8:    'uint8',
        Dtype.INT16:    'int16',
        Dtype.UINT16:   'uint16',
        Dtype.INT32:    'int32',
        Dtype.UINT32:   'uint32',
        Dtype.INT64:    'int64',
        Dtype.UINT64:   'uint64',
        Dtype.ALIGNED_INT4: 'aligned_int4',
        Dtype.ALIGNED_UINT4: 'aligned_uint4',
        Dtype.ALIGNED_INT12: 'aligned_int12',
        Dtype.ALIGNED_UINT12: 'aligned_uint12',
    }
    return dt_dict[dt]


def dtype2bits(dt):
    dt_dict = {
        Dtype.BFP16:    16,
        Dtype.FP16:     16,
        Dtype.FP32:     32,
        Dtype.FP64:     64,
        Dtype.BOOL:     8,
        Dtype.INT8:     8,
        Dtype.UINT8:    8,
        Dtype.INT16:    16,
        Dtype.UINT16:   16,
        Dtype.INT32:    32,
        Dtype.UINT32:   32,
        Dtype.INT64:    64,
        Dtype.UINT64:   64,
        Dtype.ALIGNED_INT4: 4,
        Dtype.ALIGNED_UINT4: 4,
        Dtype.ALIGNED_INT12: 12,
        Dtype.ALIGNED_UINT12: 12,
    }
    return dt_dict[dt]


def dtype2bytes(dt):
    dt_dict = {
        Dtype.BFP16:    2,
        Dtype.FP16:     2,
        Dtype.FP32:     4,
        Dtype.FP64:     8,
        Dtype.BOOL:     1,
        Dtype.INT8:     1,
        Dtype.UINT8:    1,
        Dtype.INT16:    2,
        Dtype.UINT16:   2,
        Dtype.INT32:    4,
        Dtype.UINT32:   4,
        Dtype.INT64:    8,
        Dtype.UINT64:   8,
        Dtype.ALIGNED_INT4: 1,
        Dtype.ALIGNED_UINT4: 1,
        Dtype.ALIGNED_INT12: 2,
        Dtype.ALIGNED_UINT12: 2,
    }
    return dt_dict[dt]


def dtype2tftype(dtype):
    import tensorflow.compat.v1 as tf
    tf_dict = {Dtype.BOOL: tf.bool,
               Dtype.FP16: tf.float16,
               Dtype.BFP16: tf.bfloat16,
               Dtype.FP32: tf.float32,
               Dtype.FP64: tf.float64,
               Dtype.INT16: tf.int16,
               Dtype.INT32: tf.int32,
               Dtype.INT64: tf.int64,
               Dtype.INT8: tf.int8,
               Dtype.UINT16: tf.uint16,
               Dtype.UINT32: tf.uint32,
               Dtype.UINT64: tf.uint64,
               Dtype.UINT8: tf.uint8,
               Dtype.ALIGNED_INT4: tf.int8,
               Dtype.ALIGNED_UINT4: tf.uint8,
               Dtype.ALIGNED_INT12: tf.int16,
               Dtype.ALIGNED_UINT12: tf.uint16,
               }
    return tf_dict[dtype]


def tftype2dtype(dtype):
    import tensorflow.compat.v1 as tf
    tf_dict = {tf.bool: Dtype.BOOL,
               tf.float16: Dtype.FP16,
               tf.bfloat16: Dtype.BFP16,
               tf.float32: Dtype.FP32,
               tf.float64: Dtype.FP64,
               tf.int16: Dtype.INT16,
               tf.int32: Dtype.INT32,
               tf.int64: Dtype.INT64,
               tf.int8: Dtype.INT8,
               tf.uint16: Dtype.UINT16,
               tf.uint32: Dtype.UINT32,
               tf.uint64: Dtype.UINT64,
               tf.uint8: Dtype.UINT8,
               tf.int8: Dtype.ALIGNED_INT4,
               tf.uint8: Dtype.ALIGNED_UINT4,
               tf.int16: Dtype.ALIGNED_INT12,
               tf.uint16: Dtype.ALIGNED_UINT12,
               }
    return tf_dict[dtype]


def dtype2nptype(dtype):
    import numpy as np
    np_dict = {Dtype.BOOL: np.bool8,
               Dtype.FP16: np.float16,
               Dtype.FP32: np.float32,
               Dtype.FP64: np.float64,
               Dtype.INT16: np.int16,
               Dtype.INT32: np.int32,
               Dtype.INT64: np.int64,
               Dtype.INT8: np.int8,
               Dtype.UINT16: np.uint16,
               Dtype.UINT32: np.uint32,
               Dtype.UINT64: np.uint64,
               Dtype.UINT8: np.uint8,
               Dtype.ALIGNED_INT4: np.int8,
               Dtype.ALIGNED_UINT4: np.uint8,
               Dtype.ALIGNED_INT12: np.int16,
               Dtype.ALIGNED_UINT12: np.uint16,
               }
    if Dtype.BFP16 == dtype:
        from bfloat16 import bfloat16 as np_bf16_dtype
        return np_bf16_dtype
    return np_dict[dtype]


def nptype2dtype(t):
    import numpy as np
    np_dict = {np.bool8: Dtype.BOOL,
               np.float16: Dtype.FP16,
               np.float32: Dtype.FP32,
               np.float64: Dtype.FP64,
               np.int16: Dtype.INT16,
               np.int32: Dtype.INT32,
               np.int64: Dtype.INT64,
               np.int8: Dtype.INT8,
               np.uint16: Dtype.UINT16,
               np.uint32: Dtype.UINT32,
               np.uint64: Dtype.UINT64,
               np.uint8: Dtype.UINT8,
               }
    return np_dict[t]


def nptype2torch_type(t):
    import numpy as np
    np_dict = {np.bool8: torch.bool,
               np.float16: torch.float16,
               np.float32: torch.float32,
               np.float64: torch.float64,
               np.int16: torch.int16,
               np.int32: torch.int32,
               np.int64: torch.int64,
               np.int8: torch.int8,
               np.uint16: torch.int32,
               np.uint32: torch.int64,
               np.uint64: torch.long,
               np.uint8: torch.int16,
               }
    return np_dict[t]


def bits2range(bits, is_signed):
    if is_signed:
        return -2**(bits-1), 2**(bits-1)-1
    else:
        return 0, 2**bits - 1


def range2bits(vmin, vmax, force_int=False):
    out_signed = (vmin < 0) or force_int
    out_bits = 0
    while True:
        qmin, qmax = bits2range(out_bits, out_signed)
        if vmin >= qmin and vmax <= qmax:
            break
        out_bits += 1
    return out_bits, out_signed


def dtype2range(dtype):
    bits = dtype2bits(dtype)
    signed = is_signed(dtype)
    qmin, qmax = bits2range(bits, signed)
    return qmin, qmax


def range2dtype(vmin, vmax, force_int=False):
    qbits = 8
    out_signed = True
    if vmin >= 0.0:
        out_signed = False
    out_signed = out_signed or force_int
    dtype = bits2dtype(qbits, is_signed=out_signed)
    if out_signed:
        if (vmax > 127 and vmax <= 32767) or (vmin < -128 and vmin >= -32768):
            qbits = 16
            dtype = bits2dtype(qbits, is_signed=out_signed)
        elif vmax > 32767 or vmin < -32768:
            qbits = 32
            dtype = bits2dtype(qbits, is_signed=out_signed)
    else:
        if (vmax > 255 and vmax <= 65535):
            qbits = 16
            dtype = bits2dtype(qbits, is_signed=out_signed)
        elif vmax > 65535:
            qbits = 32
            dtype = bits2dtype(qbits, is_signed=out_signed)
    return qbits, dtype


def torch_type2nptype(tp):
    import numpy as np
    th_dict = {
        torch.float16:     np.float16,
        torch.float32:     np.float32,
        torch.float64:     np.float64,
        torch.int8:        np.int8,
        torch.uint8:       np.uint8,
        torch.int16:       np.int16,
        # torch.uint16:      False,
        torch.int32:       np.int32,
        # torch.uint32:      False,
        torch.int64:       np.int64,
        # torch.uint64:      False,
        torch.bool:        np.bool8,
        torch.long:        np.int64,
        torch.int:         np.int32,
    }
    return th_dict[tp]


def torch_type2dtype(tp):
    th_dict = {
        torch.bfloat16:    Dtype.BFP16,
        torch.float16:     Dtype.FP16,
        torch.float32:     Dtype.FP32,
        torch.float64:     Dtype.FP64,
        torch.int8:        Dtype.INT8,
        torch.uint8:       Dtype.UINT8,
        torch.int16:       Dtype.INT16,
        # torch.uint16:      False,
        torch.int32:       Dtype.INT32,
        # torch.uint32:      False,
        torch.int64:       Dtype.INT64,
        # torch.uint64:      False,
        torch.bool:        Dtype.UINT8,
        torch.long:        Dtype.INT64,
        torch.int:         Dtype.INT32,
    }
    return th_dict[tp]


def dtype2torch_type(tp):
    th_dict = {
        Dtype.BFP16:        torch.bfloat16,
        Dtype.FP16:         torch.float16,
        Dtype.FP32:         torch.float32,
        Dtype.FP64:         torch.float64,
        Dtype.INT8:         torch.int8,
        Dtype.UINT8:        torch.uint8,
        Dtype.INT16:        torch.int16,
        Dtype.UINT16:       torch.int32,
        Dtype.INT32:        torch.int32,
        Dtype.UINT32:       torch.int64,
        Dtype.INT64:        torch.int64,
        Dtype.ALIGNED_INT4: torch.int8,
        Dtype.ALIGNED_UINT4: torch.int8,
        Dtype.ALIGNED_INT12: torch.int16,
        Dtype.ALIGNED_UINT12: torch.int16,
    }
    return th_dict[tp]
