# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *

'''
layer_id=26
layer_name=yolo3_tiny_regionfuse
layer_type=RegionFuse
layer_bottom=[yolo3_tiny_region_1_0,yolo3_tiny_region_2_0,yolo3_tiny_region_1_1,yolo3_tiny_region_2_1,yolo3_tiny_region_1_2,
              yolo3_tiny_region_2_2,yolo3_tiny_region_1_3,yolo3_tiny_region_2_3,yolo3_tiny_region_1_4,yolo3_tiny_region_2_4]
layer_bottom_shape=[[1,5000],[1,5000],[1,5000,4],[1,5000,4],[1,80],[1,80],[1,80],[1,80],[1,1],[1,1]]
layer_bottom_type=[float32,float32,float32,float32,float32,float32,float32,float32,float32,float32]
layer_top=[yolo3_tiny_region_3_0,yolo3_tiny_region_3_1,yolo3_tiny_region_3_2,yolo3_tiny_region_3_3,yolo3_tiny_region_3_4]
layer_top_shape=[[1,10000],[1,10000,4],[1,80],[1,80],[1,1]]
layer_top_type=[float32,float32,float32,float32,float32]
'''


@op_register(OpType.RegionFuse)
def regionfuse(self, *args):
    dev = self.inputs[0].betensor.device
    inp_score0, inp_score1 = self.inputs[0].betensor, self.inputs[1].betensor
    inp_box0, inp_box1 = self.inputs[2].betensor, self.inputs[3].betensor
    inp_num_perclass0, inp_num_perclass1 = self.inputs[4].betensor, self.inputs[5].betensor
    inp_label0, inp_label1 = self.inputs[6].betensor, self.inputs[7].betensor
    inp_total_class_num0, inp_total_clss_num1 = self.inputs[8].betensor, self.inputs[9].betensor

    batch_size = inp_score0.shape[0]
    score_shape = [batch_size] + list(self.outputs[0].ir_shape[1:])
    box_shape = [batch_size] + list(self.outputs[1].ir_shape[1:])
    num_perclass_shape = [batch_size] + list(self.outputs[2].ir_shape[1:])
    label_shape = [batch_size] + list(self.outputs[3].ir_shape[1:])
    out_score = torch.zeros(*score_shape, device=dev)
    out_box = torch.zeros(*box_shape, device=dev)
    out_num_perclass = torch.zeros(*num_perclass_shape, device=dev)
    out_label = torch.zeros(*label_shape, device=dev)
    out_total_class_num = torch.zeros(*self.inputs[8].betensor.shape, device=dev)
    if self.quantized:
        norm_score_scale = self.params['score_scale_value']
        norm_score_shift = self.params['score_shift_value']
        norm_box_scale = self.params['box_scale_value']
        norm_box_shift = self.params['box_shift_value']

        # inp_score0 = torch.round((inp_score0 * norm_score_scale[0]).long() * 0.5 ** norm_score_shift[0])
        # inp_score1 = torch.round((inp_score1 * norm_score_scale[1]).long() * 0.5 ** norm_score_shift[1])
        # inp_box0 = torch.round((inp_box0 * norm_box_scale[0]).long() * 0.5 ** norm_box_shift[0])
        # inp_box1 = torch.round((inp_box1 * norm_box_scale[1]).long() * 0.5 ** norm_box_shift[1])
        inp_score0 = ((inp_score0 * norm_score_scale[0]).long() >> norm_score_shift[0])
        inp_score1 = ((inp_score1 * norm_score_scale[1]).long() >> norm_score_shift[1])
        inp_box0 = ((inp_box0 * norm_box_scale[0]).long() >> norm_box_shift[0])
        inp_box1 = ((inp_box1 * norm_box_scale[1]).long() >> norm_box_shift[1])
        if self.quantized:
            inp_score0_qmin, inp_score0_qmax = dtype2range(self.inputs[0].dtype)
            inp_score1_qmin, inp_score1_qmax = dtype2range(self.inputs[1].dtype)
            inp_box0_qmin, inp_box0_qmax = dtype2range(self.inputs[2].dtype)
            inp_box1_qmin, inp_box1_qmax = dtype2range(self.inputs[3].dtype)
            inp_score0 = torch.clamp(inp_score0, inp_score0_qmin, inp_score0_qmax)
            inp_score1 = torch.clamp(inp_score1, inp_score1_qmin, inp_score1_qmax)
            inp_box0 = torch.clamp(inp_box0, inp_box0_qmin, inp_box0_qmax)
            inp_box1 = torch.clamp(inp_box1, inp_box1_qmin, inp_box1_qmax)

    for b in range(batch_size):
        # ob: one batch
        inp_ob_score0, inp_ob_score1 = inp_score0[b], inp_score1[b]
        inp_ob_box0, inp_ob_box1 = inp_box0[b], inp_box1[b]
        inp_ob_num_perclass0, inp_ob_num_perclass1 = inp_num_perclass0[b], inp_num_perclass1[b]
        inp_ob_label0, inp_ob_label1 = inp_label0[b], inp_label1[b]
        inp_ob_total_class_num0, inp_ob_total_class_num1 = inp_total_class_num0[b], inp_total_clss_num1[b]

        act_total_class_num0 = inp_ob_total_class_num0.int().item()
        act_total_class_num1 = inp_ob_total_class_num1.int().item()
        all_labels = torch.cat([inp_ob_label0[:act_total_class_num0], inp_ob_label1[:act_total_class_num1]])
        inp_ob_num_perclass0 = inp_ob_num_perclass0[:act_total_class_num0]
        inp_ob_num_perclass1 = inp_ob_num_perclass1[:act_total_class_num1]
        inp_ob_label0, inp_ob_label1 = inp_ob_label0[:act_total_class_num0], inp_ob_label1[:act_total_class_num1]
        fuse_labels, fuse_labels_num = torch.unique(all_labels, return_counts=True)
        fuse_total_class_num = fuse_labels.numel()
        num0_start = torch.cumsum(inp_ob_num_perclass0, dim=0) - inp_ob_num_perclass0
        num1_start = torch.cumsum(inp_ob_num_perclass1, dim=0) - inp_ob_num_perclass1

        out_label[b][0:fuse_total_class_num] = fuse_labels
        out_total_class_num[b] = torch.tensor(fuse_total_class_num, device=dev)
        fuse_ps = 0
        for i, fuse_label in enumerate(fuse_labels):
            # only has 1 and 2 in fuse_labels_num
            if fuse_labels_num[i] == 1:
                if fuse_label in inp_ob_label0:
                    num_perclass0 = inp_ob_num_perclass0[inp_ob_label0 == fuse_label].int().item()
                    start_ps0 = num0_start[inp_ob_label0 == fuse_label].int().item()
                    out_score[b][fuse_ps:fuse_ps +
                                 num_perclass0] = (inp_ob_score0[start_ps0:start_ps0 + num_perclass0])
                    out_box[b][fuse_ps:fuse_ps + num_perclass0,
                               :] = (inp_ob_box0[start_ps0:start_ps0 + num_perclass0, :])
                    fuse_ps += num_perclass0

                    out_num_perclass[b][i] = num_perclass0
                else:
                    num_perclass1 = inp_ob_num_perclass1[inp_ob_label1 == fuse_label].int().item()
                    start_ps1 = num1_start[inp_ob_label1 == fuse_label].int().item()
                    out_score[b][fuse_ps:fuse_ps+num_perclass1] = (inp_ob_score1[start_ps1:start_ps1+num_perclass1])
                    out_box[b][fuse_ps:fuse_ps + num_perclass1,
                               :] = (inp_ob_box1[start_ps1:start_ps1 + num_perclass1, :])
                    fuse_ps += num_perclass1

                    out_num_perclass[b][i] = num_perclass1
            else:
                num_perclass0 = inp_ob_num_perclass0[inp_ob_label0 == fuse_label].int().item()
                num_perclass1 = inp_ob_num_perclass1[inp_ob_label1 == fuse_label].int().item()
                start_ps0 = num0_start[inp_ob_label0 == fuse_label].int().item()
                start_ps1 = num1_start[inp_ob_label1 == fuse_label].int().item()

                out_score[b][fuse_ps:fuse_ps+num_perclass0] = (inp_ob_score0[start_ps0:start_ps0+num_perclass0])
                out_box[b][fuse_ps:fuse_ps+num_perclass0, :] = (inp_ob_box0[start_ps0:start_ps0+num_perclass0, :])
                fuse_ps += num_perclass0
                out_score[b][fuse_ps:fuse_ps+num_perclass1] = (inp_ob_score1[start_ps1:start_ps1+num_perclass1])
                out_box[b][fuse_ps:fuse_ps+num_perclass1, :] = (inp_ob_box1[start_ps1:start_ps1+num_perclass1, :])
                fuse_ps += num_perclass1
                out_num_perclass[b][i] = num_perclass0 + num_perclass1

        self.outputs[0].betensor = out_score
        self.outputs[1].betensor = out_box
        self.outputs[2].betensor = out_num_perclass
        self.outputs[3].betensor = out_label
        self.outputs[4].betensor = out_total_class_num
        return [o.betensor for o in self.outputs]


@quant_register(OpType.RegionFuse)
def regionfuse_quantize(self, *args):

    inp0, inp1, inp2, inp3 = self.inputs[:4]
    # for class score scale
    # score_scales = [inp0.scale, inp1.scale]
    # min_score_scale = min(*score_scales)
    ss = 2**inp0.qbits - 1
    norm_score_scales = [ss, ss]
    norm_score_shifts = [inp0.qbits, inp0.qbits]
    # for box coord scale
    # max value of input box is 2**inp.scale from region output
    box_scales = [inp2.scale, inp3.scale]
    min_box_scale = min(*box_scales)
    norm_box_scales = [min_box_scale / s * 4096 for s in box_scales]
    reduce_sft = (inp2.qbits-is_signed(inp2.dtype)-torch.floor(torch.log2(torch.tensor(min_box_scale))).int().item())
    norm_box_shifts = [12-reduce_sft, 12-reduce_sft]

    self.params['score_scale_value'] = norm_score_scales
    self.params['score_shift_value'] = norm_score_shifts
    self.params['box_scale_value'] = norm_box_scales
    self.params['box_shift_value'] = norm_box_shifts
    self.params['score_scale_type'] = [bits2dtype(inp0.qbits, is_signed=False)] * 2
    self.params['score_shift_type'] = [SHIFT_DTYPE] * 2
    self.params['box_scale_type'] = [bits2dtype(inp2.qbits, is_signed=False)] * 2
    self.params['box_shift_type'] = [SHIFT_DTYPE] * 2

    inp4, inp5, inp6, inp7, inp8, inp9 = self.inputs[4:]
    _id_to_inps = {
        0: [inp0, inp1],
        1: [inp2, inp3],
        2: [inp4, inp5],
        3: [inp6, inp7],
        4: [inp8, inp9],
    }

    for i, o in enumerate(self.outputs):
        in_scales = [inp.scale for inp in _id_to_inps[i]]
        min_in_scales, min_index = min(*in_scales), in_scales.index(min(*in_scales))
        o.scale = min_in_scales
        o.zerop = 0.
        o.qbits = _id_to_inps[i][min_index].qbits
        o.dtype = _id_to_inps[i][min_index].dtype
        o.qmin, o.qmax = dtype2range(o.dtype)
        o.qinvariant = _id_to_inps[i][min_index].qinvariant

    self.outputs[0].scale = 2**inp0.qbits - 1
    self.outputs[1].scale = self.outputs[1].qmax
