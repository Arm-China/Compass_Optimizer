# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3
from enum import Enum, unique, auto

__all__ = [
    "Dtype",
    "OpType",
    "register_optype",
]


class AutoName(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name


@unique
class Dtype(AutoName):
    BFP16 = auto()
    FP16 = auto()
    FP32 = auto()
    FP64 = auto()
    BOOL = auto()
    INT8 = auto()
    UINT8 = auto()
    INT16 = auto()
    UINT16 = auto()
    INT32 = auto()
    UINT32 = auto()
    INT64 = auto()
    UINT64 = auto()
    ALIGNED_INT4 = auto()
    ALIGNED_UINT4 = auto()
    ALIGNED_INT12 = auto()
    ALIGNED_UINT12 = auto()


class _OpType:
    def __init__(self):
        pass


OpType = _OpType()


class OpTypeValue(str):
    pass


OpTypeValue.name = property(lambda self: str(self[7:]), None)


def register_optype(t: str):
    global OpType
    if t not in OpType.__dict__:
        OpType.__setattr__(t, OpTypeValue("OpType."+t))
    return OpType.__getattribute__(t)


register_optype('Abs')
register_optype('AccidentalHits')
register_optype('Acos')
register_optype('Acosh')
register_optype('Activation')
register_optype('Add')
register_optype('ArgMinMax')
register_optype('Asin')
register_optype('Asinh')
register_optype('BNLL')
register_optype('BasicLSTM')
register_optype('BatchNorm')
register_optype('BatchToDepth')
register_optype('BatchToSpace')
register_optype('BiasAdd')
register_optype('BitShift')
register_optype('BoundingBox')
register_optype('CELU')
register_optype('CRELU')
register_optype('CTCBeamDecoder')
register_optype('CTCGreedyDecoder')
register_optype('Cast')
register_optype('Ceil')
register_optype('ChannelShuffle')
register_optype('Clip')
register_optype('Compress')
register_optype('Concat')
register_optype('Constant')
register_optype('ConvInteger')
register_optype('ConvTranspose')
register_optype('ConvTranspose3D')
register_optype('Convolution')
register_optype('Convolution3D')
register_optype('Cosh')
register_optype('Cosine')
register_optype('Count')
register_optype('Crop')
register_optype('CropAndResize')
register_optype('DataStride')
register_optype('DeQuantize')
register_optype('DecodeBox')
register_optype('DepthToSpace')
register_optype('DepthwiseConv')
register_optype('DetectionOutput')
register_optype('Div')
register_optype('ELU')
register_optype('Eltwise')
register_optype('Erf')
register_optype('Exp')
register_optype('FakeQuantWithMinMaxVars')
register_optype('Filter')
register_optype('Floor')
register_optype('FractionalPool')
register_optype('FullyConnected')
register_optype('GELU')
register_optype('GRUv1')
register_optype('GRUv3')
register_optype('Gather')
register_optype('GatherElements')
register_optype('GatherND')
register_optype('Gemm')
register_optype('GenerateProposals')
register_optype('GridSample')
register_optype('GroupNorm')
register_optype('HardSigmoid')
register_optype('Hardmax')
register_optype('Hardswish')
register_optype('InTopK')
register_optype('Input')
register_optype('InstanceNorm')
register_optype('Interp')
register_optype('LRN')
register_optype('LayerNorm')
register_optype('LeakyRELU')
register_optype('Log')
register_optype('LogSoftmax')
register_optype('Logical')
register_optype('MISH')
register_optype('MVN')
register_optype('MatMul')
register_optype('MatMulInteger')
register_optype('MaxPoolingWithArgMax')
register_optype('MaxRoiPool')
register_optype('MaxUnpool')
register_optype('Meshgrid')
register_optype('Mod')
register_optype('Moments')
register_optype('Mul')
register_optype('NMS')
register_optype('Negative')
register_optype('Normalization')
register_optype('OneHot')
register_optype('OverlapAdd')
register_optype('PRELU')
register_optype('Pad')
register_optype('Permute')
register_optype('Pooling')
register_optype('Pooling3D')
register_optype('PostNMS1')
register_optype('PostNMS2')
register_optype('Pow')
register_optype('Proposal')
register_optype('PyramidROIAlign')
register_optype('Quantize')
register_optype('RELU')
register_optype('RELU6')
register_optype('ROIPooling')
register_optype('Reciprocal')
register_optype('Reduce')
register_optype('Region')
register_optype('RegionFuse')
register_optype('Repeat')
register_optype('Reshape')
register_optype('Resize')
register_optype('ReverseSequence')
register_optype('RgbToYuv')
register_optype('RNN')
register_optype('RoiAlign')
register_optype('Round')
register_optype('Rsqrt')
register_optype('SELU')
register_optype('SHRINK')
register_optype('ScatterElements')
register_optype('ScatterND')
register_optype('SegmentReduce')
register_optype('Sigmoid')
register_optype('Sign')
register_optype('Silu')
register_optype('Sine')
register_optype('Sinh')
register_optype('Slice')
register_optype('Softmax')
register_optype('Softplus')
register_optype('Softsign')
register_optype('Sort')
register_optype('SpaceToBatch')
register_optype('SpaceToDepth')
register_optype('Split')
register_optype('Sqrt')
register_optype('Square')
register_optype('SquaredDifference')
register_optype('Squeeze')
register_optype('StridedSlice')
register_optype('Sub')
register_optype('THRESHOLDEDRELU')
register_optype('Tan')
register_optype('Tanh')
register_optype('Tile')
register_optype('TopK')
register_optype('Transpose')
register_optype('UpsampleByIndex')
register_optype('Where')
register_optype('YuvToRgb')
register_optype('ZeroFraction')
