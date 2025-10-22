# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from ..utils import is_match
from .concat_fuser import ConcatFusion
from .convolution_fuser import ConvBNActFusion
from .eltwise_fuser import MulFusion, AddFusion
from .expand_fuser import ExpandFusion
from .fullyconnected_fuser import LinearBNActFusion
from .gelu_fuser import GeLUFusion
from .hardswish_fuser import HardswishFusion
from .hardsigmoid_fuser import HardsigmoidFusion
from .layernorm_fuser import LayerNormFusion
from .matmul_bn_fuser import MatMulBatchNormFusion
from .multiheadattention_fuser import MultiheadAttentionFusion
from .mha_fuser import MHAFusion
from .pooling_fuser import AvgPool2dFusion, MaxPool2dFusion
from .reshape_fuser import ReshapeFusion
from .transpose_fuser import TransposeFusion
