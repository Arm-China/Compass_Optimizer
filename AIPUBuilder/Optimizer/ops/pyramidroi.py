# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *


register_optype('PyramidROIAlign')

# log2 for pyramid roialign, for it is not general, we use it just here


def mylog2(Am):
    logtab0 = [-32768, -28262, -24149, -20365, -16862, -13600, -10549, -7683, -4981, -2425]
    logtab1 = [22529, 20567, 18920, 17517, 16308, 15255, 14330, 13511, 12780, 12124]  # / *delta_y0 / delta_x0: Q13 * /
    logtab2 = [16384, 18022, 19660, 21299, 22938, 24576, 26214, 27853, 29491, 31130]  # / *x0: Q15 * /

    if Am == 0:
        return -32768
    else:
        point = 0
        var = Am
        while var < 16384:
            point = point+1
            var = var << 1
    Am = (Am << point)
    point1 = (-point) * 512  # / *inputQ15, output Q9 * /
    index = (int(Am - 16384) * 20) >> 15  # / *tableindex * /
    dx = int(Am - logtab2[index])
    dy = (dx * logtab1[index]) >> 13
    y = (dy + logtab0[index]) >> 6  # / *Q9 * /
    y = point1 + y  # /*log2(x), Q9 * /
    return y


@op_register(OpType.PyramidROIAlign)
def PyramidROIAlign(self, *args):
    # get each level's area range
    if not self.quantized:
        L1 = 0.09565604811391672
        L2 = 0.02391401202847918
        L2_h = 0.09565604811391672
        L3 = 0.005978503007119795
        L3_h = 0.02391401202847918
        L4 = 0.005978503007119795
    else:
        L1_Q = 102703631
        L2_Q = 25675908
        L2_Q_h = 102703631
        L3_Q = 6418977
        L3_Q_h = 25675908
        L4_Q = 6418977
    out = self.outputs[0].betensor
    nor_box = self.inputs[0].betensor + (torch.tensor(0) if not self.quantized else torch.tensor(self.inputs[0].zerop))
    feature_maps = []
    for i, inp in enumerate(self.inputs):
        # (torch.tensor(0) if not self.quantized else torch.tensor(inp.zerop))
        feature_maps.append(inp.betensor + (torch.tensor(0) if not self.quantized else torch.tensor(inp.zerop)))
    resize_height_ = self.outputs[0].ir_shape[1]
    resize_width_ = self.outputs[0].ir_shape[2]
    channel_ = self.outputs[0].ir_shape[3]

    y0, x0, y1, x1 = nor_box[0, :, 0], nor_box[0, :, 1], nor_box[0, :, 2], nor_box[0, :, 3]
    h = y1 - y0
    w = x1 - x0
    # Use shape of first image. Images in a batch must have the same size.
    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    dev = self.inputs[0].betensor.device
    if self.quantized:
        qvalue = self.get_param('box_input_qvalue')
        qmax = (1 << qvalue)-1
        # #quantized forward
        # image_area = (round((10 - 7.807) * 512) ) # 10 is log2(1024*1024), 7.807 is log2(224)
        # roi_level =torch.zeros((1000),dtype=torch.int16)
        # for i in range(nor_box.shape[1]):
        #     area = (h[i].int()*w[i].int())>>15
        #     area = area&0x7fff #to ensure area is postive
        #     level = ((image_area  + (mylog2(area)) // 2)+256)>>9
        #     level = min(5, max(
        #         2, 4 + (round(level))))
        #     roi_level[i]=level

        roi_level5 = torch.ones_like(h)*5
        roi_level4 = torch.ones_like(h)*4
        roi_level3 = torch.ones_like(h)*3
        roi_level2 = torch.ones_like(h)*2
        area = h.int()*w.int()
        area = area & 0x7fffFFFF
        f5 = (area > L1_Q)*roi_level5
        f4 = ((area < L2_Q_h) & (area > L2_Q))*roi_level4
        f3 = ((area < L3_Q_h) & (area > L3_Q))*roi_level3
        f2 = (area < L4_Q) * roi_level2
        roi_level = (f2+f3+f4+f5).int()
    else:
        # float32 forward
        # image_area = 10 #10 is log2(1024*1024), 7.807 is log2(224)
        # roi_level = image_area-7.807+ torch.log2(h * w)/2
        # roi_level = torch.minimum(torch.tensor(5).to(device), torch.maximum(
        #     torch.tensor(2).to(device), 4 + torch.round(roi_level).int()))
        roi_level5 = torch.ones_like(h)*5
        roi_level4 = torch.ones_like(h)*4
        roi_level3 = torch.ones_like(h)*3
        roi_level2 = torch.ones_like(h)*2
        area = h*w
        f5 = (area > L1)*roi_level5
        f4 = ((area < L2_h) & (area > L2))*roi_level4
        f3 = ((area < L3_h) & (area > L3))*roi_level3
        f2 = (area < L4) * roi_level2
        roi_level = (f2+f3+f4+f5).int()

    # the roialign algorithm
    if self.quantized:
        resize_feature = torch.zeros((nor_box.shape[1], resize_height_, resize_width_, channel_), device=dev)
        # for batchidx in range(feature.shape[0]):
        batch_idx = 0
        for boxidx in range(nor_box.shape[1]):
            feature = feature_maps[roi_level[boxidx] - 1]

            image_width = feature.shape[1]
            image_height = feature.shape[2]
            channel = feature.shape[3]

            y0_q = nor_box[batch_idx, boxidx, 0].int()
            x0_q = nor_box[batch_idx, boxidx, 1].int()
            y1_q = nor_box[batch_idx, boxidx, 2].int()
            x1_q = nor_box[batch_idx, boxidx, 3].int()
            height_scale_q = (torch.div((y1_q - y0_q) * (image_height - 1) * 256,
                              (resize_height_ - 1), rounding_mode='trunc')) >> 8  # Q15*Q0
            width_scale_q = (torch.div((x1_q - x0_q) * (image_width - 1) * 256,
                             (resize_width_ - 1), rounding_mode='trunc')) >> 8

            x_q = (image_width-1) * x0_q + torch.arange(0, resize_width_, device=out.device) * width_scale_q
            y_q = (image_height-1) * y0_q + torch.arange(0, resize_height_, device=out.device) * height_scale_q

            yy_q = torch.clamp(y_q, 0,  (image_width - 1)*qmax)
            xx_q = torch.clamp(x_q, 0, (image_height - 1)*qmax)

            top_y_index_q = yy_q >> qvalue
            bottom_y_index_q = (yy_q+qmax) >> qvalue
            y_lerp_q = (yy_q & 0x7fff).reshape(resize_height_, 1).repeat(1, channel)
            # y_lerp = torch.repeat(y_lerp, channel).reshape(resize_height_, channel)
            left_x_index_q = xx_q >> qvalue
            right_x_index_q = (xx_q+qmax) >> qvalue
            x_lerp_q = (xx_q & 0x7fff).reshape(resize_width_, 1).repeat(1, channel)

            for idxh in range(resize_height_):
                for idxw in range(resize_width_):
                    # get 4 point
                    top_left = feature[0, top_y_index_q[idxh], left_x_index_q[idxw], :]  # Q12
                    top_right = feature[0, top_y_index_q[idxh], right_x_index_q[idxw], :]  # Q22
                    bottom_left = feature[0, bottom_y_index_q[idxh], left_x_index_q[idxw], :]  # Q11
                    bottom_right = feature[0, bottom_y_index_q[idxh], right_x_index_q[idxw], :]  # Q21

                    # bilinear interpretate
                    # f(x,y)=Q12*(1-x_lerp)*(1-y_lerp)+Q22*x_lerp*(1-y_lerp)+Q11*(1-x_lerp)*y_lerp+Q21*x_lerp*y_lerp
                    # Q11=bottom_left
                    # Q12=top_left
                    # Q21=bottom_right
                    # Q22=top_right
                    # data=top_left*(1-x_lerp[idxw,:])*(1-y_lerp[idxh,:])+top_right*(1-y_lerp[idxh,:])*x_lerp[idxw,:]
                    # data=data+bottom_left*(1-x_lerp[idxw,:])*y_lerp[idxh,:]+bottom_right*x_lerp[idxw,:]*y_lerp[idxh,:]
                    # xy=y_lerp[idxh,:]*x_lerp[idxw,:]
                    # fourpoint_sum=(top_left+bottom_right-top_right-bottom_left)*xy
                    # top = top_left + (top_right-top_left)* x_lerp[idxw,:]
                    # bottom=(bottom_left - top_left) * y_lerp[idxh,:]
                    # data = fourpoint_sum+top+bottom

                    xy_q = y_lerp_q[idxh, :]*x_lerp_q[idxw, :] >> qvalue
                    fourpoint_sum = (top_left+bottom_right-top_right-bottom_left)*xy_q
                    top = (top_left.long() << qvalue) + ((top_right-top_left) * x_lerp_q[idxw, :])
                    bottom = (bottom_left - top_left) * y_lerp_q[idxh, :]
                    data_q = (fourpoint_sum+top+bottom).long() >> qvalue

                    resize_feature[boxidx, idxh, idxw, :] = data_q
    else:
        resize_feature = torch.zeros((nor_box.shape[1], resize_height_, resize_width_, channel_), device=dev)
        # for batchidx in range(feature.shape[0]):
        batch_idx = 0
        for boxidx in range(nor_box.shape[1]):
            feature = feature_maps[roi_level[boxidx] - 1]

            y0 = nor_box[batch_idx, boxidx, 0]
            x0 = nor_box[batch_idx, boxidx, 1]
            y1 = nor_box[batch_idx, boxidx, 2]
            x1 = nor_box[batch_idx, boxidx, 3]
            image_width = feature.shape[1]
            image_height = feature.shape[2]
            channel = feature.shape[3]

            height_scale = (y1 - y0) * (image_height - 1) / (resize_height_ - 1)
            width_scale = (x1 - x0) * (image_width - 1) / (resize_width_ - 1)

            x = (image_width-1) * x0 + torch.arange(0, resize_width_, device=out.device) * width_scale
            y = (image_height-1) * y0 + torch.arange(0, resize_height_, device=out.device) * height_scale

            yy = torch.clamp(y, 0,  image_width - 1)
            xx = torch.clamp(x, 0, image_height - 1)
            top_y_index = (torch.floor(yy)).int()
            bottom_y_index = (torch.ceil(yy)).int()
            y_lerp = (yy - top_y_index).reshape(resize_height_, 1).repeat(1, channel)
            # y_lerp = torch.repeat(y_lerp, channel).reshape(resize_height_, channel)
            left_x_index = (torch.floor(xx)).int()
            right_x_index = (torch.ceil(xx)).int()
            x_lerp = (xx - left_x_index).reshape(resize_width_, 1).repeat(1, channel)
            # x_lerp=torch.repeat(x_lerp,channel).reshape(resize_width_,channel)

            for idxh in range(resize_height_):
                for idxw in range(resize_width_):
                    # get 4 point
                    top_left = feature[0, top_y_index[idxh], left_x_index[idxw], :]  # Q12
                    top_right = feature[0, top_y_index[idxh], right_x_index[idxw], :]  # Q22
                    bottom_left = feature[0, bottom_y_index[idxh], left_x_index[idxw], :]  # Q11
                    bottom_right = feature[0, bottom_y_index[idxh], right_x_index[idxw], :]  # Q21

                    # bilinear interpretate
                    # f(x,y)=Q12*(1-x_lerp)*(1-y_lerp)+Q22*x_lerp*(1-y_lerp)+Q11*(1-x_lerp)*y_lerp+Q21*x_lerp*y_lerp
                    # Q11=bottom_left
                    # Q12=top_left
                    # Q21=bottom_right
                    # Q22=top_right
                    # data=top_left*(1-x_lerp[idxw,:])*(1-y_lerp[idxh,:])+top_right*(1-y_lerp[idxh,:])*x_lerp[idxw,:]
                    # data=data+bottom_left*(1-x_lerp[idxw,:])*y_lerp[idxh,:]+bottom_right*x_lerp[idxw,:]*y_lerp[idxh,:]
                    xy = y_lerp[idxh, :]*x_lerp[idxw, :]
                    fourpoint_sum = (top_left+bottom_right-top_right-bottom_left)*xy
                    top = top_left + (top_right-top_left) * x_lerp[idxw, :]
                    bottom = (bottom_left - top_left) * y_lerp[idxh, :]
                    data = fourpoint_sum+top+bottom

                    resize_feature[boxidx, idxh, idxw, :] = data

    self.outputs[0].betensor = resize_feature
    return self.outputs[:]


@quant_register(OpType.PyramidROIAlign)
def PyramidROIAlign_quantize(self, *args):
    inp = self.inputs[1]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = 0  # output has been symetric
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    self.params['box_input_qvalue'] = torch.round(torch.log2(torch.tensor(self.inputs[0].scale).float())).int().item()
