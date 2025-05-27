# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.
import datetime
import numpy as np
import torch.nn as nn
import torch

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *


class ArcReactor(object):
    FractionsInG = 0
    FractionsInA = 1
    FractionsInB = 2
    FractionsInF = 3

    @staticmethod
    def F23():
        """
        F(2,3)
        """
        BT = np.array([
            [1,  0, -1,  0],
            [0,  1,  1,  0],
            [0, -1,  1,  0],
            [0,  1,  0, -1]
        ]).astype(np.float32)

        G = np.array([
            [1,      0,   0],
            [1/2,  1/2, 1/2],
            [1/2, -1/2, 1/2],
            [0,      0,   1]
        ]).astype(np.float32)

        AT = np.array([
            [1,  1,  1,  0],
            [0,  1, -1, -1]
        ]).astype(np.float32)

        return BT, G, AT

    @classmethod
    def run(cls, m, r):
        if m == 2 and r == 3:
            return cls.F23()


class WinogradChecker(object):
    __WinogradConfig = {
        "WinogradChecked":     {},
        "Fmnrs":               [2, 3],
        "dims":                "1D",
        "DDRUtilRatio":        0.8,
        "TargetList":          ["V3C4",       "V3B4"],
        "WinogradLayerList":   {"V3C4": [],    "V3B4": []},
        "WinogradLayerFull":   {"V3C4": False, "V3B4": False},
    }
    __CPK = {
        "Wino": [60, 110],
        "Conv": [36,  72],
    }

    __CS = {
        "V3B4": {
            "Wino": {
                "Cin":  None,
                "Cout":   16,
                "H":       2,
                "W":       8
            },
            "Conv": {
                "Cin":   None,
                "Cout":    16,
                "H":        1,
                "W":        8
            }
        },
        "V3C4": {
            "Wino": {
                "Cin":   None,
                "Cout":    16,
                "H":        4,
                "W":        8
            },
            "Conv": {
                "Cin":   None,
                "Cout":    16,
                "H":        2,
                "W":        8
            }
        }
    }

    def _checker(self, Target, node):
        if node.type != OpType.Convolution or 'with_winograd' not in node.attrs.keys() or not node.attrs['with_winograd']:
            return False

        # PadedInputWidth = node.inputs[0].ir_shape[2] + (node.get_param('pad_left')+ node.get_param('pad_right'))
        _Cin = node.inputs[0].ir_shape[3]
        _Hin = node.inputs[0].ir_shape[1]
        _Win = node.inputs[0].ir_shape[2]
        _Cout = node.outputs[0].ir_shape[3]
        _Hout = node.outputs[0].ir_shape[1]
        _Wout = node.outputs[0].ir_shape[2]
        n, r = self.__WinogradConfig["Fmnrs"]
        m = n + r - 1

        if m != 2 and r != 3:
            return False

        WeightsPrefetchOverhead = (r*m - r*r) * _Cin * _Cout / (32 * self.__WinogradConfig["DDRUtilRatio"])
        InputChannelGounpSize = 0 if _Cin == 32 else 1
        CubeSizeNormalConv = self.__CS[Target]["Conv"]
        CubeSizeWinograd = self.__CS[Target]["Wino"]
        # according to ich_gsize get ic.
        CubeSizeNormalConv["Cin"] = (2**InputChannelGounpSize) * 32
        CubeSizeWinograd["Cin"] = (2**InputChannelGounpSize) * 32
        kernelNumNormalConv = np.ceil(_Cin/CubeSizeNormalConv['Cin']) * np.ceil(_Cout/CubeSizeNormalConv['Cout']) \
            * np.ceil(_Hout/CubeSizeNormalConv['H']) * np.ceil(_Wout/CubeSizeNormalConv['W'])
        kernelNumWinograd = np.ceil(_Cin/CubeSizeWinograd['Cin']) * np.ceil(_Cout/CubeSizeWinograd['Cout']) \
            * np.ceil(_Hout/CubeSizeWinograd['H']) * np.ceil(_Wout/CubeSizeWinograd['W'])

        PrefetchKernelCycleNomalConv = self.__CPK["Conv"][InputChannelGounpSize] * kernelNumNormalConv
        PrefetchKernelCycleWinograd = self.__CPK["Wino"][InputChannelGounpSize] * kernelNumWinograd

        RequiredRules = [
            lambda node: (node.get_param("group") == 1,
                          'group size support group = 1(%d).' % node.get_param("group")),
            lambda node: (_Cin > 4,
                          'for HW spec Cin need > 4(%d).' % _Cin),
            lambda node: (_Hin > 7 and _Win > 7,
                          'inp shape (h,w) need > (7,7)(%d,%d).' % (_Hin, _Win)),
            lambda node: (node.get_param("kernel_x") == node.get_param("kernel_y") == 3,
                          'weights shape support Kw == Kh == 3(%d,%d).' % (node.get_param("kernel_x"), node.get_param("kernel_y"))),
            lambda node: (node.get_param("stride_x") == node.get_param("stride_y") == 1,
                          'stride only support stride_x == stride_y == 1(%d,%d).' % (node.get_param("stride_x"), node.get_param("stride_y"))),
            lambda node: (node.get_param("dilation_x") == node.get_param("dilation_y") == 1,
                          'weights shape support dilation_x == dilation_y == 1(%d,%d).' % (node.get_param("dilation_x"), node.get_param("dilation_y"))),
            lambda node: (PrefetchKernelCycleWinograd + WeightsPrefetchOverhead < PrefetchKernelCycleNomalConv,
                          "winograd HW perf problem.(not recommend)")
        ]

        # stage1 rule checker.
        for rule in RequiredRules:
            RulePassed, log = rule(node)
            if not RulePassed:
                msg = log + " layer_id = " + str(node.get_attrs("layer_id")) + ", \
                      it is not recommended to use winograd in " + Target
                OPT_DEBUG(msg)
                # OPT_INFO(msg)
                return False

        # stage2 display checker.
        IsFull = WinogradChecker.AddWinogradLayerId(Target, int(node.get_attrs("layer_id")))
        if not IsFull:
            msg = "layer_id = " + str(node.get_attrs("layer_id")) + ", \
                it is recommended to use winograd(F(2,3) for default) in " + Target
            OPT_DEBUG(msg)
            # OPT_INFO(msg)
        return True

    @classmethod
    def run(cls, node, MODE="Release"):
        try:
            # for muti-target: as long as there is a non-compliance,
            # we do not choose to use winograd;
            # because we don't know what hardware the customer is using.
            # so, we may need get a Targer Item in cfg file.
            LayeridString = node.get_attrs("layer_id")
            if LayeridString in cls.__WinogradConfig["WinogradChecked"].keys():
                return cls.__WinogradConfig["WinogradChecked"][LayeridString]

            status = True
            for Target in cls.__WinogradConfig["TargetList"]:
                status = status and cls()._checker(Target, node)

            cls.__WinogradConfig["WinogradChecked"][LayeridString] = status
            node.attrs['is_winograd_checker_passed'] = status
            return status

        except Exception:
            return False

    @classmethod
    def GetWinogradLayerIds(cls):
        return cls.__WinogradConfig["WinogradLayerList"]

    @classmethod
    def SetWinogradLayerIds(cls, layer_ids):
        cls.__WinogradConfig["WinogradLayerList"] = layer_ids

    @classmethod
    def AddWinogradLayerId(cls, target, layer_id):
        """
        return: IsFull
        """
        TargetLayerIdFull = cls.__WinogradConfig["WinogradLayerFull"][target]
        if TargetLayerIdFull:
            return True

        TargetLayerIds = cls.__WinogradConfig["WinogradLayerList"][target]
        if layer_id in TargetLayerIds and TargetLayerIdFull == False:
            TargetLayerIdFull = True
            return True

        if layer_id not in TargetLayerIds:
            TargetLayerIds.append(layer_id)
            return False

    @classmethod
    def GetFmnrs(cls):
        return cls.__WinogradConfig["Fmnrs"]

    @classmethod
    def GetWinogradDims(cls):
        return cls.__WinogradConfig["dims"]

    @classmethod
    def GetWinogradKernelStride(cls):
        Fmnrs = cls.__WinogradConfig["Fmnrs"]
        dims = cls.__WinogradConfig["dims"]
        if dims == '1D':
            KernelStride = {}
            m, r = Fmnrs
            TileInWidth = m + r - 1
            TileOverlap = r - 1
            KernelStride["kernel_x"] = TileInWidth
            KernelStride["kernel_y"] = r
            KernelStride["stride_x"] = TileOverlap
            KernelStride["stride_y"] = 1
            return KernelStride

        if dims == "2D":
            KernelStride = {}
            m, n, r, s = Fmnrs
            TileInWidth = m + r - 1
            TileInHeight = n + s - 1
            TileOverlap = r - 1
            KernelStride["kernel_x"] = TileInWidth
            KernelStride["kernel_y"] = TileInHeight
            KernelStride["stride_x"] = TileOverlap
            KernelStride["stride_y"] = TileOverlap
            return KernelStride

        if dims == "3D":
            KernelStride = {}
            # Todo
            return KernelStride


def align(x, ali):
    assert x > 0, "align x must > 0."
    return (x + ali - 1) // ali * ali


def WinogradAllocator(self, inp, weights, bias, DEBUG=False):

    def GetWinogradParams(self):
        # for now HW only support 1D F(2,3)
        m, r = WinogradChecker.GetFmnrs()
        return m, r

    if DEBUG:
        padding = (self.get_param('pad_left'), self.get_param('pad_right'),
                   self.get_param('pad_top'), self.get_param('pad_bottom'))
        inp_pad = nn.functional.pad(inp, padding, value=self.inputs[0].zerop[0])
        if "OriginWeights" in self.attrs.keys():
            ori_w = self.attrs["OriginWeights"]
            ori_w = nhwc2nchw(ori_w)
        else:
            ori_w = weights
        GT = nn.functional.conv2d(inp_pad,
                                  ori_w,
                                  bias,
                                  stride=(1, 1),
                                  padding=0,
                                  dilation=(self.get_param("dilation_y"), self.get_param("dilation_x")),
                                  groups=self.get_param("group")
                                  )

        # if self.attrs["layer_id"] in ["7", '8', '11']:
        #     OPT_INFO("flag")
        StartTime = datetime.datetime.now()

    # Some strategies to determine the parameters: m,r
    m, r = GetWinogradParams(self)
    WinogradOut = winograd_conv_1D_HP(self, inp, weights, bias, m, r, DEBUG)

    if DEBUG:
        EndTime = datetime.datetime.now()
        OPT_INFO("[Winograd]layer %d winograd_conv_1D_HP spend %d micro-second." %
                 (int(self.attrs['layer_id']), (EndTime - StartTime).seconds))

        diff = GT - WinogradOut
        OPT_INFO("[conv_float - wino_float]diff max: %f" % diff.max())

    return WinogradOut


def winograd_conv_1D(self, inp, weights, bias, m=2, r=3, DEBUG=True):
    # prepare params
    Batch, _Cin, _Hin, _Win = inp.shape
    _, _Hout, _Wout,  _Cout = self.outputs[0].ir_shape
    _PadLeft, _PadRight, _PadTop, _PadBottom = (int(self.get_param('pad_left')),
                                                int(self.get_param('pad_right')),
                                                int(self.get_param('pad_top')),
                                                int(self.get_param('pad_bottom')))
    _StrideX, _StrideY = (int(self.get_param("stride_x")),
                          int(self.get_param("stride_y")))
    _, _, _KernelWidth, _KernelHeight = weights.shape

    # F(2,3)
    m = 2
    r = 3
    BT, G, AT = ArcReactor.run(m, r)
    BT = torch.tensor(BT, dtype=torch.float32, device=inp.device)
    G = torch.tensor(G, dtype=torch.float32, device=inp.device)
    AT = torch.tensor(AT, dtype=torch.float32, device=inp.device)
    TileInWidth = m + r - 1
    if 'WinogradWeights' not in self.attrs.keys():
        WinogradWeights = torch.zeros((_Cout, _Cin, _KernelHeight, TileInWidth), device=inp.device)
        ConvWeithtsReshape = torch.reshape(weights, (_Cout * _Cin * _KernelHeight, _KernelWidth))
        WinogradWeights = torch.matmul(ConvWeithtsReshape, G.permute(1, 0))
        WinogradWeights = WinogradWeights.view(_Cout, _Cin, _KernelHeight, TileInWidth)
        self.attrs['WinogradWeights'] = WinogradWeights.permute((0, 2, 3, 1))  # NHWC
        winograd_weights_betensor = self.attrs['WinogradWeights']
        winograd_weights_t = PyTensor(self.name+"/constants/WinogradWeights", winograd_weights_betensor.shape)
        winograd_weights_t.ir_shape = TensorShape(list(winograd_weights_betensor.shape))
        winograd_weights_t.betensor = winograd_weights_betensor
        self.constants['WinogradWeights'] = winograd_weights_t
    else:
        WinogradWeights = self.attrs['WinogradWeights'].permute((0, 3, 1, 2))

    WinogradPadRight = _PadRight + (align(_PadLeft + _Win + _PadRight - 4, 2) - (_PadLeft + _Win + _PadRight - 4))
    winogradPadBottom = _PadBottom
    WinogradPadTop = _PadTop
    WinogradPadLeft = _PadLeft

    WinogradWout = (_PadLeft + _Win + WinogradPadRight - _KernelWidth) // _StrideX + 1
    WinogradHout = (_PadTop + _Hin + winogradPadBottom - _KernelHeight) // _StrideY + 1
    WinogradCout = _Cout

    LoopWidth = WinogradWout // (_KernelWidth - _StrideX)
    LoopHeight = WinogradHout

    tmp = inp.clone()  # NCHW
    WinogradPaddingFunc = nn.ConstantPad2d((WinogradPadTop, winogradPadBottom, WinogradPadLeft,
                                            WinogradPadRight), value=self.inputs[0].zerop)  # need NCHW, 2d: (up, down, left, right)
    WinogradInp = WinogradPaddingFunc(tmp)

    WinogradOut = torch.zeros((1, WinogradCout, WinogradHout, WinogradWout), device=inp.device)
    WinogradTileOut1D = torch.zeros((1, WinogradCout, 1, 2), device=inp.device)

    # only support batch == 1 now.
    for k in range(WinogradCout):
        for h in range(LoopHeight):
            for w in range(LoopWidth):
                InpTileBatchxCinx3x4 = WinogradInp[:, :, h:h+3, 2*w:2 *
                                                   w+TileInWidth]  # [batch, Cin, 3,4] input tile block
                OutTile3X4 = torch.zeros((r, m+r-1), device=WinogradInp.device)  # [3,4]
                for kh in range(_KernelHeight):  # 3
                    for c in range(_Cin):
                        BT_d = torch.matmul(BT, InpTileBatchxCinx3x4[0, c, kh])  # [4,4] matmul [4,1]--> [4,]
                        OutTile3X4[kh] = torch.add(OutTile3X4[kh], WinogradWeights[k, c, kh]
                                                   * BT_d)  # [4,] eltwiseMUL [4,] --> out_unit[i,4]

                BTdGgTile = torch.sum(OutTile3X4, dim=0)  # [4,1]
                WinogradTileOut1D[0, k] = torch.matmul(AT, BTdGgTile)  # [2,4] matmul [4,1] --> [2,]
                WinogradOut[:, k:k+1, h:h+1, 2*w:2*w+2] = WinogradTileOut1D[:, k:k+1, :, :]

    if WinogradWout != _Wout:
        WinogradOut = WinogradOut[:, :, :, :_Wout]

    out = WinogradOut.add(bias.view(1, _Cout, 1, 1).repeat(1, 1, _Hout, _Wout))

    return out


def winograd_conv_1D_HP(self, inp, weights, bias, m=2, r=3, DEBUG=True):
    # prepare params
    Batch, _Cin, _Hin, _Win = inp.shape
    _, _Hout, _Wout,  _Cout = self.outputs[0].ir_shape
    _PadLeft, _PadRight, _PadTop, _PadBottom = (int(self.get_param('pad_left')),
                                                int(self.get_param('pad_right')),
                                                int(self.get_param('pad_top')),
                                                int(self.get_param('pad_bottom')))
    if self.quantized:
        _StrideX, _StrideY = (int(self.get_param("stride_y")), int(self.get_param("stride_y")))
        _KernelHeight, _KernelWidth = weights.shape[2], weights.shape[2]
    else:
        _StrideX, _StrideY = (int(self.get_param("stride_x")), int(self.get_param("stride_y")))
        _, _, _KernelHeight, _KernelWidth = weights.shape

    # F(2,3)
    BT, G, AT = ArcReactor.run(m, r)
    BT = torch.tensor(BT, dtype=torch.float32, device=inp.device)
    G = torch.tensor(G,  dtype=torch.float32, device=inp.device)
    AT = torch.tensor(AT, dtype=torch.float32, device=inp.device)
    TileInWidth = m + r - 1
    Overlap = m

    WinogradPadRight = _PadRight + (align(_PadLeft + _Win + _PadRight - 4, 2) - (_PadLeft + _Win + _PadRight - 4))
    winogradPadBottom = _PadBottom
    WinogradPadTop = _PadTop
    WinogradPadLeft = _PadLeft

    WinogradWout = (_PadLeft + _Win + WinogradPadRight - _KernelWidth) // _StrideX + 1
    WinogradHout = (_PadTop + _Hin + winogradPadBottom - _KernelHeight) // _StrideY + 1
    WinogradCout = _Cout

    LoopWidth = WinogradWout // (_KernelWidth - _StrideX)
    LoopHeight = WinogradHout

    tmp = inp.clone()  # NCHW
    WinogradPaddingFunc = nn.ConstantPad2d((WinogradPadLeft, WinogradPadRight, WinogradPadTop,
                                            winogradPadBottom), value=self.inputs[0].zerop[0])  # need NCHW, 2d: ( left,right,up, down,)
    PaddingInp = WinogradPaddingFunc(tmp)

    if 'WinogradWeights' not in self.attrs.keys() and not self.get_param('with_winograd', optional=True, default_value=False):
        # generate G*g
        WinogradWeights = torch.zeros((_Cout, _Cin, _KernelHeight, TileInWidth),
                                      device=inp.device, dtype=torch.float32)  # 64,64,3,4
        weithts1D = torch.reshape(weights.type(torch.float32), (_Cout * _Cin * _KernelHeight, _KernelWidth))
        WinogradWeights = torch.matmul(weithts1D, G.permute(1, 0))  # [*, 3] matmul [3, 4]
        WinogradWeights = WinogradWeights.view(_Cout, _Cin, _KernelHeight, TileInWidth)
        self.attrs['WinogradWeights'] = WinogradWeights.permute((0, 2, 3, 1))
        winograd_weights_betensor = self.attrs['WinogradWeights']
        winograd_weights_t = PyTensor(self.name+"/constants/WinogradWeights", winograd_weights_betensor.shape)
        winograd_weights_t.ir_shape = TensorShape(list(winograd_weights_betensor.shape))
        winograd_weights_t.betensor = winograd_weights_betensor
        self.constants['WinogradWeights'] = winograd_weights_t

    if self.quantized:
        Gg_weights = weights.float()
    else:
        Gg_weights = self.attrs['WinogradWeights'].permute((0, 3, 1, 2)).float()

    def DataReverseMethod(benchmark=False):
        data_reverse = torch.zeros((Batch, _Cin, LoopHeight, LoopWidth, _KernelHeight, TileInWidth), device=inp.device)

        def method0(benchmark):
            # method1 warm up.
            # reorder input for compute on GPU
            if benchmark:
                StartTime = datetime.datetime.now()
            # tmp = torch.zeros((Batch, _Cin, LoopHeight, LoopWidth, _KernelHeight, TileInWidth), device=inp.device) #64,58*28,3,4)
            for b in range(Batch):
                for h in range(LoopHeight):             # 56, 7
                    for w in range(LoopWidth):          # 28, 2
                        for _kh in range(_KernelHeight):  # 3,
                            for c in range(_Cin):       # 512,7,4,3,4
                                data_reverse[b, c, h, w, _kh, :] = PaddingInp[b, c, h+_kh, 2*w:2*w+TileInWidth]
            if benchmark:
                EndTime = datetime.datetime.now()
                OPT_INFO("[Torch]layer %d reverse input spend t0[warm up] %d micro-second." %
                         (int(self.attrs['layer_id']), (EndTime - StartTime).microseconds))
            return data_reverse

        def method1(benchmark):
            if benchmark:
                StartTime = datetime.datetime.now()
            # tmp = torch.zeros((Batch, _Cin, LoopHeight, LoopWidth, _KernelHeight, TileInWidth), device=inp.device)
            for b in range(Batch):
                for h in range(LoopHeight):             # 56, 7
                    for w in range(LoopWidth):          # 28, 2
                        for _kh in range(_KernelHeight):  # 3,
                            for c in range(_Cin):       # 512,7,4,3,4
                                data_reverse[b, c, h, w, _kh, :] = PaddingInp[b, c,  h+_kh, 2*w:2*w+TileInWidth]
            if benchmark:
                EndTime = datetime.datetime.now()
                OPT_INFO("[Torch]layer %d reverse input spend t1[loop] %d micro-second." %
                         (int(self.attrs['layer_id']), (EndTime - StartTime).microseconds))
            return data_reverse

        def method2(benchmark):
            # numpy method.
            if benchmark:
                StartTime = datetime.datetime.now()
            np_data_reverse = np.zeros((Batch, _Cin, LoopHeight, LoopWidth, _KernelHeight, TileInWidth))
            np_inp_padding = PaddingInp.cpu().numpy()
            for b in range(Batch):
                for h in range(LoopHeight):             # 56, 7
                    for w in range(LoopWidth):          # 28, 2
                        for _kh in range(_KernelHeight):  # 3,
                            for c in range(_Cin):       # 512,7,4,3,4
                                np_data_reverse[b, c, h, w, _kh, :] = np_inp_padding[b, c, h+_kh, 2*w:2*w+TileInWidth]
            gt = torch.tensor(np_data_reverse, dtype=torch.float32, device=inp.device)
            if benchmark:
                EndTime = datetime.datetime.now()
                OPT_INFO("[numpy]layer %d reverse input spend t2[numpy] %d micro-second." %
                         (int(self.attrs['layer_id']), (EndTime - StartTime).microseconds))
            return gt

        def method3(benchmark):
            if benchmark:
                StartTime = datetime.datetime.now()
            w_slice = torch.split(PaddingInp, 2, dim=3)  # [1,64,58,58] --> (29*[1,64,58,2])
            double_part = list(w_slice[1:-1])  # tuple(tensor) 29-->27
            double_part2d = [[d]*2 for d in double_part]
            double_part1d = [x for d2 in double_part2d for x in d2]
            totalW_part = [w_slice[0]] + double_part1d + [w_slice[-1]]
            expanded_W = torch.cat(totalW_part, dim=3)

            h_sclice = torch.split(expanded_W, 1, dim=2)
            tmp = []
            for i, _slice in enumerate(h_sclice):
                if i < 2:
                    continue
                H3Wtotal = torch.cat((h_sclice[i-2], h_sclice[i-1], h_sclice[i]),
                                     dim=2)  # [B,C_in,3,W_expanded]: [1,64,3,112]
                H3W4_tuple = torch.split(H3Wtotal, 4, dim=3)  # tuple(28*[1,64,3,4])
                Wstack_H3W4 = torch.stack(H3W4_tuple, dim=2)  # [1, 64, 28, 3, 4]
                tmp.append(Wstack_H3W4)

            data_reverse = torch.stack(tmp, dim=2)
            if benchmark:
                EndTime = datetime.datetime.now()
                OPT_INFO("[Torch]layer %d reverse input spend t3[hp] %d micro-second." %
                         (int(self.attrs['layer_id']), (EndTime - StartTime).microseconds))
            return data_reverse

        if benchmark:
            _ = method0(benchmark)
            _ = method1(benchmark)
            GT = method2(benchmark)
            _out = method3(benchmark)
        else:
            _out = method3(benchmark)
        return _out.float()

    if DEBUG:
        StartTime = datetime.datetime.now()

    DataReverse = DataReverseMethod(benchmark=DEBUG)

    if DEBUG:
        EndTime = datetime.datetime.now()
        OPT_INFO("layer %d input data reverse[optimized method] spend %d micro-second." %
                 (int(self.attrs['layer_id']), (EndTime - StartTime).microseconds))

    WinogradOut = torch.zeros((Batch, _Cout, LoopHeight, int(LoopWidth * 2)), device=inp.device, dtype=torch.float32)

    def WinogradKernels(benchmark=False):
        def isBatchMergeMode():
            DtyepeSize = 4  # Byte
            InputSize = Batch * _Cin * _Hin * _Win * DtyepeSize
            TiledInputSize = Batch * _Cin * LoopHeight * LoopWidth * _KernelHeight * TileInWidth * DtyepeSize
            max_size = TiledInputSize * _Cout
            if max_size > 1024*1024*1024:
                return False
            return True

        def BatchMergeMode(benchmark):
            if not isBatchMergeMode():
                OPT_INFO("not support batch merge mode.")
                return 0

            if benchmark:
                StartTime = datetime.datetime.now()

            # compute AT * (G * g + BT * d)
            data_reverse = torch.matmul(DataReverse, BT.permute(1, 0))
            BT_out = data_reverse.unsqueeze(1).repeat((1, _Cout, 1, 1, 1, 1, 1))
            winograd_weights_new = torch.reshape(Gg_weights, (_Cout, _Cin, 1, 1, _KernelHeight, TileInWidth))
            winograd_weights_new = winograd_weights_new.repeat((1, 1, LoopHeight, LoopWidth, 1, 1))
            BT_out = BT_out * winograd_weights_new
            BT_out = torch.sum(BT_out, dim=2)
            BT_out = torch.sum(BT_out, dim=4)
            AT_out_1 = torch.matmul(BT_out, AT.permute(1, 0))  # 64,56,28,2
            WinogradOut = torch.reshape(AT_out_1, (Batch, _Cout, LoopHeight, int(LoopWidth * 2)))

            if benchmark:
                EndTime = datetime.datetime.now()
                OPT_INFO("layer %d WinogradKernels[BatchMergeMode] spend %d micro-second." %
                         (int(self.attrs['layer_id']), (EndTime - StartTime).microseconds))
                # OPT_INFO("batch merge mode: enough GPU memory. ")
                # max_size = _Cout * Batch * _Cin * LoopHeight * LoopWidth * _KernelHeight * TileInWidth * 4 # 4: float32,4byte.
                # OPT_INFO("max memory footprint size of single variance[*3]: %d MByte"%(int(max_size/(1024*1024))))

        def ExchangeBatchCoutMode(benchmark):
            ################################ exchange(Batch, output_channel) Mode: cause batch can control##################
            if benchmark:
                StartTime = datetime.datetime.now()
            data_tmp = torch.matmul(DataReverse, BT.permute(1, 0))

            for o in range(_Cout):
                winograd_weights_new = torch.reshape(Gg_weights[o], (1, _Cin, 1, 1, _KernelHeight, TileInWidth))
                winograd_weights_new = winograd_weights_new.repeat((Batch, 1, LoopHeight, LoopWidth, 1, 1))
                BT_out = data_tmp * winograd_weights_new
                BT_out = torch.sum(BT_out, dim=1)
                BT_out = torch.sum(BT_out, dim=3)
                AT_out_1 = torch.matmul(BT_out, AT.permute(1, 0))  # 10,56,28,2
                WinogradOut[:, o, :, :] = torch.reshape(AT_out_1, (Batch, LoopHeight, int(LoopWidth * 2)))  # 10,56,28,2

            if benchmark:
                EndTime = datetime.datetime.now()
                OPT_INFO("layer %d WinogradKernels[ExchangeBatchCoutMode] spend %d micro-second." %
                         (int(self.attrs['layer_id']), (EndTime - StartTime).microseconds))

        def HeavyFootprintMode(benchmark):
            ################################# Split Batch Mode(heavy footprint mode)####################################
            if benchmark:
                StartTime = datetime.datetime.now()
            for b in range(Batch):
                # compute AT * (G * g + BT * d)
                data_tmp = torch.matmul(DataReverse[b], BT.permute(1, 0))
                BT_out = data_tmp.repeat((_Cout, 1, 1, 1, 1, 1))
                winograd_weights_new = torch.reshape(Gg_weights, (_Cout, _Cin, 1, 1, _KernelHeight, TileInWidth))
                winograd_weights_new = winograd_weights_new.repeat((1, 1, LoopHeight, LoopWidth, 1, 1))
                BT_out = BT_out * winograd_weights_new
                BT_out = torch.sum(BT_out, dim=1)
                BT_out = torch.sum(BT_out, dim=3)
                AT_out_1 = torch.matmul(BT_out, AT.permute(1, 0))  # 64,56,28,2
                WinogradOut[b] = torch.reshape(AT_out_1, (_Cout, LoopHeight, int(LoopWidth * 2)))  # 64,56,28,2
            if benchmark:
                EndTime = datetime.datetime.now()
                OPT_INFO("layer %d WinogradKernels[HeavyFootprintMode] %d micro-second." %
                         (int(self.attrs['layer_id']), (EndTime - StartTime).microseconds))

        def LightFootprintMode(benchmark):
            #################################Split Batch/output_channel(light footprint) mode.##########################
            if benchmark:
                StartTime = datetime.datetime.now()
            for b in range(Batch):
                data_tmp = torch.matmul(DataReverse[b], BT.permute(1, 0))
                for o in range(_Cout):
                    winograd_weights_new = torch.reshape(Gg_weights[o], (_Cin, 1, 1, _KernelHeight, TileInWidth))
                    winograd_weights_new = winograd_weights_new.repeat((1, LoopHeight, LoopWidth, 1, 1))
                    BT_out = data_tmp * winograd_weights_new
                    BT_out = torch.sum(BT_out, dim=0)
                    BT_out = torch.sum(BT_out, dim=2)
                    AT_out_1 = torch.matmul(BT_out, AT.permute(1, 0))  # 64,56,28,2
                    WinogradOut[b][o] = torch.reshape(AT_out_1, (LoopHeight, int(LoopWidth * 2)))  # 64,56,28,2
                # torch.cuda.empty_cache()
            if benchmark:
                EndTime = datetime.datetime.now()
                OPT_INFO("layer %d WinogradKernels[LightFootprintMode] %d micro-second." %
                         (int(self.attrs['layer_id']), (EndTime - StartTime).microseconds))
        if benchmark:
            BatchMergeMode(benchmark)
            LightFootprintMode(benchmark)
            HeavyFootprintMode(benchmark)
            ExchangeBatchCoutMode(benchmark)
        else:
            try:
                HeavyFootprintMode(benchmark)
            except Exception as e:
                OPT_INFO("GPU memory limited.")
                ExchangeBatchCoutMode(benchmark)

    WinogradKernels(benchmark=DEBUG)

    if WinogradWout != _Wout or WinogradHout != _Hout:
        WinogradOut = WinogradOut[:, :, :_Hout, :_Wout]

    if DEBUG:
        StartTime = datetime.datetime.now()

    out = WinogradOut.add(bias.view(1, _Cout, 1, 1).repeat(1, 1, _Hout, _Wout))

    if DEBUG:
        EndTime = datetime.datetime.now()
        OPT_INFO("[HP_PLUS]add bias spend %d micro-second." % (EndTime - StartTime).microseconds)

    m, r = WinogradChecker.GetFmnrs()
    WinogradKernelStride = WinogradChecker.GetWinogradKernelStride()
    if self.quantized:
        self.params["with_winograd"] = True
        self.params["Fmnrs"] = [m, r]
        self.params["kernel_x"] = WinogradKernelStride["kernel_x"]  # 4
        self.params["kernel_y"] = WinogradKernelStride["kernel_y"]  # 3
        self.params["stride_x"] = WinogradKernelStride["stride_x"]  # 2
        self.params["stride_y"] = WinogradKernelStride["stride_y"]  # 1

    return out


def winograd_conv_2D(self, inp, weights, bias, Fmnrs, DEBUG=True):
    # prepare params
    Batch, _Cin, _Hin, _Win = inp.shape
    _, _Hout, _Wout,  _Cout = self.outputs[0].ir_shape
    _PadLeft, _PadRight, _PadTop, _PadBottom = (int(self.get_param('pad_left')),
                                                int(self.get_param('pad_right')),
                                                int(self.get_param('pad_top')),
                                                int(self.get_param('pad_bottom')))
    _StrideX, _StrideY = (int(self.get_param("stride_x")),
                          int(self.get_param("stride_y")))
    _, _, _KernelWidth, _KernelHeight = weights.shape

    m, r = WinogradChecker.GetFmnrs()
    BT, G, AT = ArcReactor.run(m, r)
    BT = torch.tensor(BT, dtype=torch.float32, device=inp.device)
    G = torch.tensor(G,  dtype=torch.float32, device=inp.device)
    AT = torch.tensor(AT, dtype=torch.float32, device=inp.device)
    TileInWidth = m + r - 1
    TileInHeight = m + r - 1
    if 'WinogradWeights' not in self.attrs.keys():
        WinogradWeights = torch.zeros((_Cout, _Cin, TileInWidth, TileInWidth), device=inp.device)
        for k in range(_Cout):
            for c in range(_Cin):
                WinogradWeights[k, c] = torch.matmul(torch.matmul(G, weights[k, c]), G.permute(1, 0))
        self.attrs['WinogradWeights'] = WinogradWeights.permute((0, 2, 3, 1))  # NHWC
        winograd_weights_betensor = self.attrs['WinogradWeights']
        winograd_weights_t = PyTensor(self.name+"/constants/WinogradWeights", winograd_weights_betensor.shape)
        winograd_weights_t.ir_shape = TensorShape(list(winograd_weights_betensor.shape))
        winograd_weights_t.betensor = winograd_weights_betensor
        self.constants['WinogradWeights'] = winograd_weights_t
    else:
        WinogradWeights = self.attrs['WinogradWeights'].permute((0, 3, 1, 2))

    WinogradPadRight = _PadRight + (align(_PadLeft + _Win + _PadRight - 4, 2) - (_PadLeft + _Win + _PadRight - 4))
    winogradPadBottom = _PadBottom + (align(_PadTop + _Hin + _PadBottom - 4, 2) - (_PadTop + _Hin + _PadBottom - 4))
    WinogradPadTop = _PadTop
    WinogradPadLeft = _PadLeft

    WinogradWout = (_PadLeft + _Win + WinogradPadRight - _KernelWidth) // _StrideX + 1
    WinogradHout = (_PadTop + _Hin + winogradPadBottom - _KernelHeight) // _StrideY + 1
    WinogradCout = _Cout

    LoopWidth = WinogradWout // (_KernelWidth - _StrideX)
    LoopHeight = WinogradHout // (_KernelHeight - _StrideY)

    tmp = inp.clone()  # NCHW
    WinogradPaddingFunc = nn.ConstantPad2d((WinogradPadTop, winogradPadBottom, WinogradPadLeft,
                                            WinogradPadRight), value=self.inputs[0].zerop)  # need NCHW, 2d: (up, down, left, right)
    WinogradInp = WinogradPaddingFunc(tmp)

    WinogradOut = torch.zeros((1, WinogradCout, WinogradHout, WinogradWout), device=inp.device)
    WinogradTileOut2D = torch.zeros((1, WinogradCout, 2, 2), device=inp.device)

    # only support batch == 1 now.
    for k in range(WinogradCout):
        for h in range(LoopHeight):
            for w in range(LoopWidth):
                InpTileBatchxCinx4x4 = WinogradInp[:, :, 2*h:2*h+TileInHeight,
                                                   2*w:2*w+TileInWidth]  # [batch, Cin, 3,4] input tile block
                OutTile4X4 = torch.zeros((TileInHeight, TileInWidth), device=WinogradInp.device)  # [3,4]
                for c in range(_Cin):
                    # [4,4] matmul [4,1]--> [4,]
                    BT_d = torch.matmul(torch.matmul(BT, InpTileBatchxCinx4x4[0, c]), BT.permute(1, 0))
                    # [4,] eltwiseMUL [4,] --> out_unit[i,4]
                    OutTile4X4 = torch.add(OutTile4X4, WinogradWeights[k, c] * BT_d)

                WinogradTileOut2D[0, k] = torch.matmul(torch.matmul(
                    AT, OutTile4X4), AT.permute(1, 0))  # [2,4] matmul [4,1] --> [2,]
                WinogradOut[:, k:k+1, 2*h:2*h+2, 2*w:2*w+2] = WinogradTileOut2D[:, k:k+1, :, :]

    if WinogradWout != _Wout or WinogradHout != _Hout:
        WinogradOut = WinogradOut[:, :, :_Hout, :_Wout]

    out = WinogradOut.add(bias.view(1, _Cout, 1, 1).repeat(1, 1, _Hout, _Wout))
    return out
