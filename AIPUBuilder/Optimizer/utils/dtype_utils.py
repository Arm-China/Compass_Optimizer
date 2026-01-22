# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import OPT_FATAL
from typing import Union, Optional

OPT_EPSILON = torch.finfo(torch.float32).eps
OPT_INT_MIN = -2 ** 31
OPT_INT_MAX = 2 ** 31 - 1


def is_torch_tensor_with_multi_data(var):
    return True if isinstance(var, torch.Tensor) and var.numel() > 1 else False


def is_torch_tensor(var):
    return True if isinstance(var, torch.Tensor) else False


def construct_torch_tensor(var, device=None):
    try:
        dev = device if device is not None else "cpu"
        if var is None:
            constructed_t = var
        elif is_torch_tensor(var):
            constructed_t = var.to(device=dev)
        else:
            constructed_t = torch.tensor(var, device=dev)
        return constructed_t
    except Exception as e:

        OPT_WARN(f"construct torch tensor failed, and the error message is {e}, we will directly return the orignal "
                 f"input argument = {vars}")
        raise e
        # return var


torch_tensor = construct_torch_tensor


def batch_construct_torch_tensor(vars, device=None):
    if not isinstance(vars, (list, tuple)):
        vars = [vars]
    constructed_t = []
    for var in vars:
        constructed_t.append(construct_torch_tensor(var, device=device))
    return constructed_t[0] if len(constructed_t) == 1 else constructed_t


def to_list(x):
    return [x] if not isinstance(x, list) else x


def nhwc2nchw(x):
    return x.permute(0, 3, 1, 2)


def nchw2nhwc(x):
    return x.permute(0, 2, 3, 1)


def is_signed(dt):
    dt_dict = {
        Dtype.BOOL:       False,
        Dtype.FP4_E2M1FN: True,
        Dtype.FP6_E2M3FN: True,
        Dtype.FP6_E3M2FN: True,
        Dtype.FP8_E4M3FN: True,
        Dtype.FP8_E5M2: True,
        Dtype.FP16:     True,
        Dtype.BFP16:    True,
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
        torch.bool:        False,
        torch.float8_e4m3fn: True,
        torch.float8_e5m2: True,
        torch.bfloat16:    True,
        torch.float16:     True,
        torch.float32:     True,
        torch.float64:     True,
        torch.int8:        True,
        torch.uint8:       False,
        torch.int16:       True,
        torch.uint16:      False,
        torch.int32:       True,
        torch.uint32:      False,
        torch.int64:       True,
        torch.uint64:      False,
        torch.long:        True,
        torch.int:         True,
    }
    if dt in dt_dict:
        return dt_dict[dt]
    else:
        return th_dict[dt]


def is_float(dt):
    dt_dict = {
        Dtype.BOOL:       False,
        Dtype.FP4_E2M1FN: True,
        Dtype.FP6_E2M3FN: True,
        Dtype.FP6_E3M2FN: True,
        Dtype.FP8_E4M3FN: True,
        Dtype.FP8_E5M2: True,
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
        torch.bool:        False,
        torch.float8_e4m3fn: True,
        torch.float8_e5m2: True,
        torch.bfloat16:    True,
        torch.float16:     True,
        torch.float32:     True,
        torch.float64:     True,
        torch.int8:        False,
        torch.uint8:       False,
        torch.int16:       False,
        torch.uint16:      False,
        torch.int32:       False,
        torch.uint32:      False,
        torch.int64:       False,
        torch.uint64:      False,
        torch.long:        False,
        torch.int:         False,
    }
    if dt in dt_dict:
        return dt_dict[dt]
    else:
        return th_dict[dt]


def is_quant_fpx(dt):
    dt_dict = {
        Dtype.BOOL:       False,
        Dtype.FP4_E2M1FN: True,
        Dtype.FP6_E2M3FN: True,
        Dtype.FP6_E3M2FN: True,
        Dtype.FP8_E4M3FN: True,
        Dtype.FP8_E5M2: True,
        Dtype.BFP16:    False,
        Dtype.FP16:     False,
        Dtype.FP32:     False,
        Dtype.FP64:     False,
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
        torch.bool:        False,
        torch.float8_e4m3fn: True,
        torch.float8_e5m2: True,
        torch.bfloat16:    False,
        torch.float16:     False,
        torch.float32:     False,
        torch.float64:     False,
        torch.int8:        False,
        torch.uint8:       False,
        torch.int16:       False,
        torch.uint16:      False,
        torch.int32:       False,
        torch.uint32:      False,
        torch.int64:       False,
        torch.uint64:      False,
        torch.long:        False,
        torch.int:         False,
    }
    if dt in dt_dict:
        return dt_dict[dt]
    else:
        return th_dict[dt]


def is_quant_dtype(dt):
    return is_quant_fpx(dt) or (not is_float(dt))


def has_inf(dt):
    dt_dict = {
        Dtype.BOOL:       False,
        Dtype.FP4_E2M1FN: False,
        Dtype.FP6_E2M3FN: False,
        Dtype.FP6_E3M2FN: False,
        Dtype.FP8_E4M3FN: False,
        Dtype.FP8_E5M2: True,
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
        torch.bool:        False,
        torch.float8_e4m3fn: False,
        torch.float8_e5m2: True,
        torch.bfloat16:    True,
        torch.float16:     True,
        torch.float32:     True,
        torch.float64:     True,
        torch.int8:        False,
        torch.uint8:       False,
        torch.int16:       False,
        torch.uint16:      False,
        torch.int32:       False,
        torch.uint32:      False,
        torch.int64:       False,
        torch.uint64:      False,
        torch.long:        False,
        torch.int:         False,
    }
    if dt in dt_dict:
        return dt_dict[dt]
    else:
        return th_dict[dt]


def bits2dtype(bits, is_signed, use_float=False, mantissa_bits=0):
    if bits <= 4:
        if use_float:
            return Dtype.FP4_E2M1FN
        else:
            return Dtype.ALIGNED_INT4 if is_signed else Dtype.ALIGNED_UINT4
    elif bits <= 6:
        if use_float:
            if mantissa_bits >= 3:
                return Dtype.FP6_E2M3FN
            else:
                return Dtype.FP6_E3M2FN
        else:
            return Dtype.INT8 if is_signed else Dtype.UINT8
    elif bits <= 8:
        if use_float:
            if mantissa_bits >= 3:
                return Dtype.FP8_E4M3FN
            else:
                return Dtype.FP8_E5M2
        else:
            return Dtype.INT8 if is_signed else Dtype.UINT8
    elif bits <= 12:
        if use_float:
            if mantissa_bits >= 10:
                return Dtype.FP16
            else:
                return Dtype.BFP16
        else:
            return Dtype.ALIGNED_INT12 if is_signed else Dtype.ALIGNED_UINT12
    elif bits <= 16:
        if use_float:
            if mantissa_bits >= 10:
                return Dtype.FP16
            else:
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
    elif dname in ['float8_e5m2', 'fp8_e5m2', 'dtype.float8_e5m2', 'dtype.fp8_e5m2', 'float8e5m2', 'fp8e5m2']:
        return Dtype.FP8_E5M2
    elif dname in ['float8_e4m3fn', 'fp8_e4m3fn', 'dtype.float8_e4m3fn', 'dtype.fp8_e4m3fn', 'float8e4m3fn', 'fp8e4m3fn']:
        return Dtype.FP8_E4M3FN
    elif dname in ['float4_e2m1fn', 'fp4_e2m1fn', 'dtype.float4_e2m1fn', 'dtype.fp4_e2m1fn', 'float4e2m1fn', 'fp4e2m1fn']:
        return Dtype.FP4_E2M1FN
    elif dname in ['float6_e2m3fn', 'fp6_e2m3fn', 'dtype.float6_e2m3fn', 'dtype.fp6_e2m3fn', 'float6e2m3fn', 'fp6e2m3fn', ]:
        return Dtype.FP6_E2M3FN
    elif dname in ['float6_e3m2fn', 'fp6_e3m2fn', 'dtype.float6_e3m2fn', 'dtype.fp6_e3m2fn', 'float6e3m2fn', 'fp6e3m2fn', ]:
        return Dtype.FP6_E3M2FN
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
        Dtype.FP4_E2M1FN: 'float4_e2m1fn',
        Dtype.FP6_E2M3FN: 'float6_e2m1fn',
        Dtype.FP6_E3M2FN: 'float6_e3m2fn',
        Dtype.FP8_E5M2:   'float8_e5m2',
        Dtype.FP8_E4M3FN: 'float8_e4m3fn',
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
        Dtype.FP4_E2M1FN: 4,
        Dtype.FP6_E2M3FN: 6,
        Dtype.FP6_E3M2FN: 6,
        Dtype.FP8_E5M2:   8,
        Dtype.FP8_E4M3FN: 8,
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
        Dtype.FP4_E2M1FN: 1,
        Dtype.FP6_E2M3FN: 1,
        Dtype.FP6_E3M2FN: 1,
        Dtype.FP8_E5M2:   1,
        Dtype.FP8_E4M3FN: 1,
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
               Dtype.FP4_E2M1FN: tf.float16,
               Dtype.FP6_E2M3FN: tf.float16,
               Dtype.FP6_E3M2FN: tf.float16,
               Dtype.FP8_E5M2:   tf.float16,
               Dtype.FP8_E4M3FN: tf.float16,
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
    np_dict = {Dtype.BOOL: np.bool_,
               Dtype.FP4_E2M1FN: np.float16,
               Dtype.FP6_E2M3FN: np.float16,
               Dtype.FP6_E3M2FN: np.float16,
               Dtype.FP8_E5M2:   np.float16,
               Dtype.FP8_E4M3FN: np.float16,
               Dtype.BFP16:  np.float32,
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
    return np_dict[dtype]


def nptype2dtype(t):
    import numpy as np
    from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2, float6_e3m2fn, float6_e2m3fn, float4_e2m1fn
    np_dict = {np.bool_: Dtype.BOOL,
               float4_e2m1fn: Dtype.FP4_E2M1FN,
               float6_e2m3fn: Dtype.FP6_E2M3FN,
               float6_e3m2fn: Dtype.FP6_E3M2FN,
               float8_e5m2: Dtype.FP8_E5M2,
               float8_e4m3fn: Dtype.FP8_E4M3FN,
               bfloat16: Dtype.BFP16,
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
    np2dict = {}
    for k, v in np_dict.items():
        np2dict[k] = v
        np2dict[np.dtype(k)] = v

    return np2dict[t]


def nptype2torch_type(t):
    import numpy as np
    np_dict = {np.bool_: torch.bool,
               np.float16: torch.float16,
               np.float32: torch.float32,
               np.float64: torch.float64,
               np.int16: torch.int16,
               np.int32: torch.int32,
               np.int64: torch.int64,
               np.int8: torch.int8,
               np.uint16: torch.uint16,
               np.uint32: torch.uint32,
               np.uint64: torch.uint64,
               np.uint8: torch.uint8,
               }
    np2dict = {}
    for k, v in np_dict.items():
        np2dict[k] = v
        np2dict[np.dtype(k)] = v

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
    if is_float(dtype):
        import numpy as np
        if dtype in [Dtype.FP4_E2M1FN, Dtype.FP6_E2M3FN, Dtype.FP6_E3M2FN, Dtype.FP8_E4M3FN, Dtype.FP8_E5M2, Dtype.BFP16]:
            exponent, mantissa, emax = get_exponent_and_mantissa_of_dtype(dtype)
            mbits = mantissa + 2
            if dtype != Dtype.FP8_E4M3FN:
                qmax = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)
                qmin = -qmax
            else:
                qmax = 2**emax * 1.75
                qmin = -qmax
        else:
            np_dtype = dtype2nptype(dtype)
            qmin, qmax = np.finfo(np_dtype).min.item(), np.finfo(np_dtype).max.item()
    else:
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
        torch.float8_e4m3fn: np.float16,
        torch.float8_e5m2: np.float16,
        torch.bfloat16:    np.float32,
        torch.float16:     np.float16,
        torch.float32:     np.float32,
        torch.float64:     np.float64,
        torch.int8:        np.int8,
        torch.uint8:       np.uint8,
        torch.int16:       np.int16,
        torch.uint16:      np.uint16,
        torch.int32:       np.int32,
        torch.uint32:      np.uint32,
        torch.int64:       np.int64,
        torch.uint64:      np.uint64,
        torch.bool:        np.bool_,
        torch.long:        np.int64,
        torch.int:         np.int32,
    }
    return th_dict[tp]


def torch_type2dtype(tp):
    th_dict = {
        torch.float8_e4m3fn: Dtype.FP8_E4M3FN,
        torch.float8_e5m2: Dtype.FP8_E5M2,
        torch.bfloat16:    Dtype.BFP16,
        torch.float16:     Dtype.FP16,
        torch.float32:     Dtype.FP32,
        torch.float64:     Dtype.FP64,
        torch.int8:        Dtype.INT8,
        torch.uint8:       Dtype.UINT8,
        torch.int16:       Dtype.INT16,
        torch.uint16:      Dtype.UINT16,
        torch.int32:       Dtype.INT32,
        torch.uint32:      Dtype.UINT32,
        torch.int64:       Dtype.INT64,
        torch.uint64:      Dtype.UINT64,
        torch.bool:        Dtype.UINT8,
        torch.long:        Dtype.INT64,
        torch.int:         Dtype.INT32,
    }
    return th_dict[tp]


def dtype2torch_type(tp):
    th_dict = {
        Dtype.BOOL:         torch.bool,
        # torch.float8 only valid on cpu or specific gpu
        Dtype.FP4_E2M1FN:     torch.float8_e4m3fn,
        Dtype.FP6_E2M3FN:     torch.float8_e4m3fn,
        Dtype.FP6_E3M2FN:     torch.float8_e4m3fn,
        Dtype.FP4_E2M1FN:     torch.float8_e4m3fn,
        Dtype.FP8_E5M2:       torch.float8_e5m2,
        Dtype.FP8_E4M3FN:     torch.float8_e4m3fn,
        Dtype.BFP16:        torch.bfloat16,
        Dtype.FP16:         torch.float16,
        Dtype.FP32:         torch.float32,
        Dtype.FP64:         torch.float64,
        Dtype.INT8:         torch.int8,
        Dtype.UINT8:        torch.uint8,
        Dtype.INT16:        torch.int16,
        Dtype.UINT16:       torch.uint16,
        Dtype.INT32:        torch.int32,
        Dtype.UINT32:       torch.uint32,
        Dtype.INT64:        torch.int64,
        Dtype.UINT64:       torch.uint64,
        Dtype.ALIGNED_INT4: torch.int8,
        Dtype.ALIGNED_UINT4: torch.int8,
        Dtype.ALIGNED_INT12: torch.int16,
        Dtype.ALIGNED_UINT12: torch.int16,
    }
    return th_dict[tp]


def to_fp24(a: torch.Tensor):
    mask = torch.tensor(0xFFFFFF00, device=a.device).int()
    return (a.float().view(torch.int32) & mask).view(torch.float32)


def get_exponent_and_mantissa_of_dtype(dtype):
    if is_float(dtype):
        if dtype == Dtype.FP4_E2M1FN:
            exponent = 2
            mantissa = 1
            emax = 2**(exponent - 1)
        elif dtype == Dtype.FP6_E2M3FN:
            exponent = 2
            mantissa = 3
            emax = 2**(exponent - 1)
        elif dtype == Dtype.FP6_E3M2FN:
            exponent = 3
            mantissa = 2
            emax = 2**(exponent - 1)
        elif dtype == Dtype.FP8_E5M2:
            exponent = 5
            mantissa = 2
            emax = 2**(exponent - 1) - 1
        elif dtype == Dtype.FP8_E4M3FN:
            exponent = 4
            mantissa = 3
            emax = 2**(exponent - 1)
        elif dtype == Dtype.BFP16:
            exponent = 8
            mantissa = 7
            emax = 2**(exponent - 1) - 1
        elif dtype == Dtype.FP16:
            exponent = 5
            mantissa = 10
            emax = 2**(exponent - 1) - 1
        elif dtype == Dtype.FP32:
            exponent = 8
            mantissa = 23
            emax = 2**(exponent - 1) - 1
        elif dtype == Dtype.FP64:
            exponent = 11
            mantissa = 52
            emax = 2**(exponent - 1) - 1
        else:
            OPT_FATAL('unsupported dtype: %s' % dtype)
    else:
        bits = dtype2bits(dtype)
        signed = is_signed(dtype)
        mantissa = 0
        exponent = (bits - 1) if signed else bits
        emax = (bits - 1) if signed else bits
    return exponent, mantissa, emax


def get_round_mode_func(round_mode):
    round_mode = round_mode.upper()
    if round_mode == 'CEIL':
        round_func = torch.ceil
    elif round_mode == 'FLOOR':
        round_func = torch.floor
    elif round_mode == 'TRUNC':
        round_func = torch.trunc
    elif round_mode == 'NEAREST':
        def round_func(x): return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)
    else:
        round_func = torch.round
    return round_func


def get_round_func_according_to_dtype(round_mode, dtype: Optional[Dtype] = None):
    if dtype is not None and dtype in [Dtype.FP8_E4M3FN, Dtype.FP8_E5M2]:
        def round_func(x): return round_to_fp8(x, dtype, round_mode=round_mode)
    elif dtype in [Dtype.FP4_E2M1FN]:
        def round_func(x): return round_to_fp4(x, dtype, round_mode=round_mode)
    elif dtype is None or (not is_float(dtype) and dtype2bits(dtype) <= 16):
        round_func = get_round_mode_func(round_mode)
    else:
        round_func = torch.nn.Identity()
    return round_func


def fp8_to_uint8(x: torch.Tensor, dtype: Dtype, round_mode='ROUND_TO_EVEN'):
    assert dtype in [Dtype.FP8_E4M3FN, Dtype.FP8_E5M2], "dtype must in {[Dtype.FP8_E4M3FN, Dtype.FP8_E5M2]}"
    x = x.float()
    if round_mode != 'NEAREST':
        _, max_norm = dtype2range(dtype)
        private_exp = torch.floor(torch.log2(torch.abs(x) + (x == 0).type(x.dtype)))
        e_bits, m_bits, _ = get_exponent_and_mantissa_of_dtype(dtype)
        x = x / (2 ** private_exp) * (2**m_bits)
        x = get_round_mode_func(round_mode)(x)
        x = x / (2**m_bits) * (2**private_exp)
        x = torch.clamp(x, min=-max_norm, max=max_norm)
    dev = x.device
    out = torch.zeros_like(x, device=dev).int()
    nan_mask = torch.isnan(x)  # nan

    x_int32 = x[~nan_mask].float().view(torch.int32)
    valid_out = torch.zeros_like(x_int32, device=dev).int()
    sign = (x_int32 >> 31) & torch.tensor(0x1, device=x.device).int()
    exponent = ((x_int32 >> 23) & torch.tensor(0xFF, device=x.device).int()) - 127
    mantissa = x_int32 & torch.tensor(0x7FFFFF, device=x.device).int()

    if dtype == Dtype.FP8_E5M2:
        inf_mask = torch.bitwise_or(exponent > 15, torch.bitwise_and(exponent == 15, mantissa > 0x600000))
        norm_mask = torch.bitwise_and(exponent > -14, ~inf_mask)
        subnorm_mask = ~torch.bitwise_or(norm_mask, inf_mask)

        g = (mantissa[norm_mask] >> 20) & 1
        valid_out[norm_mask] = (sign[norm_mask] << 7) | (
            ((exponent[norm_mask] + 15) << 2) | ((mantissa[norm_mask] >> 21) & 0x3) + g)

        submorm_mantissa = mantissa[subnorm_mask] + (1 << 23)
        g = (submorm_mantissa >> (6 - exponent[subnorm_mask])) & 1
        valid_out[subnorm_mask] = (sign[subnorm_mask] << 7) | ((submorm_mantissa >> (7 - exponent[subnorm_mask])) + g)
        valid_out[inf_mask] = 0x7c

    else:
        nan_mask2 = torch.bitwise_or(exponent > 8, torch.bitwise_and(exponent == 8, mantissa > 0x600000))
        norm_mask = torch.bitwise_and(exponent > -7, ~nan_mask2)
        subnorm_mask = ~torch.bitwise_or(norm_mask, nan_mask2)

        g = (mantissa[norm_mask] >> 19) & 1
        valid_out[norm_mask] = (sign[norm_mask] << 7) | (
            ((exponent[norm_mask] + 7) << 3) | ((mantissa[norm_mask] >> 20) & 0x7) + g)

        submorm_mantissa = mantissa[subnorm_mask] + (1 << 23)
        g = (submorm_mantissa >> (13 - exponent[subnorm_mask])) & 1
        valid_out[subnorm_mask] = (sign[subnorm_mask] << 7) | ((submorm_mantissa >> (14 - exponent[subnorm_mask])) + g)
        valid_out[nan_mask2] = 0x7f

    out[nan_mask] = 0x7f
    out[~nan_mask] = valid_out
    out = out.to(torch.uint8)

    return out


def uint8_to_fp8(x: torch.Tensor, dtype: Dtype):
    assert dtype in [Dtype.FP8_E4M3FN, Dtype.FP8_E5M2], "dtype must in {[Dtype.FP8_E4M3FN, Dtype.FP8_E5M2]}"
    dev = x.device
    uint8_x = x.to(torch.uint8)
    out = torch.zeros_like(uint8_x, device=dev)
    if dtype == Dtype.FP8_E4M3FN:
        lookup_table = torch.tensor([0.0, 0.001953125, 0.00390625, 0.005859375, 0.0078125, 0.009765625, 0.01171875,
                                     0.013671875, 0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375, 0.025390625, 0.02734375,
                                     0.029296875, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375,
                                     0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.140625,
                                     0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625,
                                     0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375, 1.5,
                                     1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
                                     8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0,
                                     36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0, 128.0,
                                     144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, float('nan')], device=dev).float()

    else:
        lookup_table = torch.tensor([0.0, 1.52587890625e-05, 3.0517578125e-05, 4.57763671875e-05, 6.103515625e-05, 7.62939453125e-05, 9.1552734375e-05,
                                     0.0001068115234375, 0.0001220703125, 0.000152587890625, 0.00018310546875, 0.000213623046875,
                                     0.000244140625, 0.00030517578125, 0.0003662109375, 0.00042724609375, 0.00048828125, 0.0006103515625,
                                     0.000732421875, 0.0008544921875, 0.0009765625, 0.001220703125, 0.00146484375, 0.001708984375, 0.001953125,
                                     0.00244140625, 0.0029296875, 0.00341796875, 0.00390625, 0.0048828125, 0.005859375, 0.0068359375, 0.0078125,
                                     0.009765625, 0.01171875, 0.013671875, 0.015625, 0.01953125, 0.0234375, 0.02734375, 0.03125, 0.0390625, 0.046875,
                                     0.0546875, 0.0625, 0.078125, 0.09375, 0.109375, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.3125, 0.375, 0.4375,
                                     0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0,
                                     16.0, 20.0, 24.0, 28.0, 32.0, 40.0, 48.0, 56.0, 64.0, 80.0, 96.0, 112.0, 128.0, 160.0, 192.0, 224.0, 256.0,
                                     320.0, 384.0, 448.0, 512.0, 640.0, 768.0, 896.0, 1024.0, 1280.0, 1536.0, 1792.0, 2048.0, 2560.0, 3072.0,
                                     3584.0, 4096.0, 5120.0, 6144.0, 7168.0, 8192.0, 10240.0, 12288.0, 14336.0, 16384.0, 20480.0, 24576.0,
                                     28672.0, 32768.0, 40960.0, 49152.0, 57344.0, float('inf'), float('nan'), float('nan'), float('nan')], device=dev).float()
    index = uint8_x & 0x7f
    lut = lookup_table[torch.reshape(index, (-1,)).long()]
    lut = torch.reshape(lut, index.shape)
    out = torch.where((uint8_x >> 7) == 0, lut, -lut)
    return out


def fp32_to_fp4(x: torch.tensor, round_mode='ROUND_TO_EVEN'):
    # current only support round_mode = 'ROUND_TO_EVEN'
    signed = torch.sign(x)
    abs_x = torch.abs(x.float())
    abs_x = torch.where((abs_x > 0) & (abs_x <= 0.25), 0, abs_x)
    abs_x = torch.where((abs_x > 0.25) & (abs_x <= 0.75), 0.5, abs_x)
    abs_x = torch.where((abs_x > 0.75) & (abs_x <= 1.25), 1.0, abs_x)
    abs_x = torch.where((abs_x > 1.25) & (abs_x < 1.75), 1.5, abs_x)
    abs_x = torch.where((abs_x >= 1.75) & (abs_x <= 2.5), 2.0, abs_x)
    abs_x = torch.where((abs_x > 2.5) & (abs_x < 3.5), 3.0, abs_x)
    abs_x = torch.where((abs_x >= 3.5) & (abs_x < 5.0), 4.0, abs_x)
    abs_x = torch.where((abs_x >= 5.0),  6.0, abs_x)
    fp4_value = abs_x * signed
    return fp4_value


def uint8_to_fp4(x: torch.Tensor):
    uint8_x = x.to(torch.uint8)
    lookup_table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device).float()
    index = uint8_x & 0x07
    lut = lookup_table[torch.reshape(index, (-1,)).long()]
    lut = torch.reshape(lut, index.shape)
    out = torch.where((uint8_x >> 3) == 0, lut, -lut)
    return out


def fp4_to_uint4(x: torch.Tensor):
    fp4_uint4_table = [(0.0, 0b00000000), (0.5, 0b00000001), (1.0, 0b00000010), (1.5, 0b00000011),
                       (2.0, 0b00000100), (3.0, 0b00000101), (4.0, 0b00000110), (6.0, 0b00000111)]
    fp4_value = fp32_to_fp4(x)
    signed = torch.sign(fp4_value)
    fp4_value_abs = torch.abs(fp4_value)
    uint4_value = torch.zeros_like(x, device=x.device).to(torch.uint8)
    for fp4_v, uint4_v in fp4_uint4_table:
        mask = fp4_value_abs == fp4_v
        uint4_value[mask] = torch.tensor(uint4_v, device=x.device).to(torch.uint8)
    uint4_value = torch.where(signed == -1, torch.bitwise_or(uint4_value,
                              torch.tensor(0b00001000, device=x.device)).to(torch.uint8), uint4_value)
    return uint4_value


def round_to_fp8(a: torch.Tensor, dtype: Dtype, round_mode='ROUND_TO_EVEN'):
    assert dtype in [Dtype.FP8_E4M3FN, Dtype.FP8_E5M2], "dtype must in {[Dtype.FP8_E4M3FN, Dtype.FP8_E5M2]}"
    uint8_value = fp8_to_uint8(a, dtype, round_mode=round_mode)
    fp8_value = uint8_to_fp8(uint8_value, dtype)
    _, max_norm = dtype2range(dtype)
    fp8_value = torch.where(torch.isnan(fp8_value), 0, fp8_value)
    fp8_value = torch.where(torch.isinf(fp8_value), max_norm, fp8_value)
    return fp8_value


def round_to_fp4(a: torch.Tensor, dtype: Dtype, round_mode='ROUND_TO_EVEN'):
    assert dtype in [Dtype.FP4_E2M1FN], "dtype must in {[Dtype.FP4_E2M1FN]}"
    fp4_value = fp32_to_fp4(a, round_mode=round_mode)
    return fp4_value
