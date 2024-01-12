# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.plugins.aipubt_metric_mAP import BasemAPMetric
from AIPUBuilder.Optimizer.logger import *
import torch
import numpy as np
import torchvision
from scipy import special
from collections import defaultdict


@register_plugin(PluginType.Metric, '1.0')
class YOLOVOCmAPMetric(BasemAPMetric):
    def __init__(self, num_class=90):
        super().__init__()
        self.num_class = int(num_class)

    def combine_predict_label_and_extract_obj_all_class(self, batch_idx, class_list, box_list, score_list, target):
        predict_dict = {}
        targets_dict = {}
        predict_dict.update({'label_index': class_list})
        predict_dict.update({'bbox': box_list})
        predict_dict.update({'confidence': score_list})
        predict_dict.update({'image_name': target['image_name'][batch_idx]})
        for k, v in target.items():
            targets_dict.update({k: v[batch_idx].cpu().numpy()})
        BasemAPMetric.extract_obj_all_class(predict_dict, self.predicts)
        BasemAPMetric.extract_obj_all_class(targets_dict, self.targets)

    def __call__(self, pred, target):
        assert len(pred) == 9, OPT_FATAL(
            'please check the outputs number(should be 9)')
        batch = pred[2].shape[0]

        pred_list = []
        for pd in pred:
            pred_list.append(pd.cpu().numpy())
        for i in range(batch):
            predict = defaultdict()
            obj_class_num = pred_list[4][i]
            label_pre_class = pred_list[3][i]
            boxes = pred_list[5][i]
            box_num_pre_class = pred_list[6][i]
            scores = pred_list[7][i]
            keep = pred_list[8][i]

            label_id_list = []
            boxes_list = []
            score_list = []
            all_box_idx = 0
            for cid in range(int(obj_class_num[0])):
                box_num_cur_class = int(box_num_pre_class[cid])
                label_id = label_pre_class[cid]

                label_id_list.extend([label_id]*box_num_cur_class)
                boxes_list.extend([boxes[all_box_idx+box_idx]
                                   for box_idx in range(box_num_cur_class)])
                score_list.extend([scores[all_box_idx+box_idx]
                                   for box_idx in range(box_num_cur_class)])

                all_box_idx = all_box_idx + box_num_cur_class
            self.combine_predict_label_and_extract_obj_all_class(i, label_id_list, boxes_list, score_list, target)

    def reset(self):
        super().reset()

    def compute(self):

        self.eval_mAP(self.predicts, self.targets, self.num_class, 0)
        return self.mAP

    def report(self):
        return "Yolo mAP accuracy is %f" % (self.compute())


@register_plugin(PluginType.Metric, '1.0')
class YOLOV1tinyVOCmAPMetric(YOLOVOCmAPMetric):
    """
    This YOLOV1tinyVOCmAPMetric is used for the metric of yolov1_tiny model in Optimizer.

    The input image size of facebox model is 300x300.
    score_threshold=0.25, iou_threshold=0.45
    """

    def __call__(self, pred, target):
        batch = pred[0].shape[0]

        pred_list = []
        for pd in pred:
            pred_list.append(pd.cpu().numpy())

        for i in range(batch):
            label_id_list, boxes_list, score_list = self.v1_decode_output(
                pred_list[0][i])
            self.combine_predict_label_and_extract_obj_all_class(i, label_id_list, boxes_list, score_list, target)

    def v1_decode_output(self, output, threshold=0.2, iou_threshold=0.5, feature_length=7, class_num=20):
        class_offset = feature_length * feature_length * class_num
        scales_offset = feature_length * feature_length * 2
        probs = np.zeros((feature_length, feature_length, 2, class_num))
        class_probs = np.reshape(output[:class_offset], (feature_length, feature_length, class_num))
        scales = np.reshape(output[class_offset:class_offset+scales_offset], (feature_length, feature_length, 2))
        boxes = np.reshape(output[class_offset+scales_offset:], (feature_length, feature_length, 2, 4))

        offset = np.transpose(np.reshape(
            np.array([np.arange(feature_length)]*(feature_length*2)), (2, feature_length, feature_length)), (1, 2, 0))

        boxes[..., 0] += offset
        boxes[..., 1] += np.transpose(offset, (1, 0, 2))
        boxes[..., 0:2] = boxes[..., 0:2] / float(feature_length)
        boxes[..., 2] = np.multiply(boxes[..., 2], boxes[..., 2])
        boxes[..., 3] = np.multiply(boxes[..., 3], boxes[..., 3])

        for i in range(2):
            for j in range(class_num):
                probs[..., i, j] = np.multiply(
                    class_probs[..., j], scales[..., i])
        filter_mat_probs = np.array(probs >= threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i+1, len(boxes_filtered)):
                if BasemAPMetric.cal_iou_xywh(boxes_filtered[i], boxes_filtered[j]) > iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]
        boxes_list = []
        for box in boxes_filtered:
            x = box[0]
            y = box[1]
            w = box[2]/2
            h = box[3]/2
            boxes_list.append([y-h, x-w, y+h, x+w])

        return classes_num_filtered, boxes_list, probs_filtered


@register_plugin(PluginType.Metric, '1.0')
class YOLOV5onnxmAPMetric(YOLOVOCmAPMetric):
    """
    This YOLOV5cocomAPMetric is used for the metric of yolov5_onnx model in Optimizer.

    The input image size of facebox model is 300x300.
    score_threshold=0.25, iou_threshold=0.45
    """

    def __call__(self, pred, target):
        batch = pred[0].shape[0]

        def xywh2yxyx(x):
            # Convert boxes from [x, y, w, h] to [xmin, ymin, xmax, ymax]
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[..., 1] = x[..., 0] - x[..., 2] / 2
            y[..., 0] = x[..., 1] - x[..., 3] / 2
            y[..., 3] = x[..., 0] + x[..., 2] / 2
            y[..., 2] = x[..., 1] + x[..., 3] / 2
            return y

        for i in range(batch):
            label_id_list, boxes_list, score_list = self.decode_output(
                pred[i], coordinate_convert_func=xywh2yxyx)
            self.combine_predict_label_and_extract_obj_all_class(i, label_id_list, boxes_list, score_list, target)

    def decode_output(self, prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False, multi_label=False, max_det=300, coordinate_convert_func=None):
        class_num = prediction.shape[2] - 5  # number of classes
        confidence = prediction[..., 4] > conf_thres

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        min_wh, max_wh = 2, 7680
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        redundant = True  # require redundant detections
        multi_label &= class_num > 1  # multiple labels per box (adds 0.5ms/img)

        for xi, pred in enumerate(prediction):  # image index, image inference
            pred = pred[confidence[xi]]
            if not pred.shape[0]:
                return [], [], []

            # Compute conf
            pred[..., 5:] *= pred[..., 4:5]  # conf = obj_conf * cls_conf

            if coordinate_convert_func is None:
                OPT_FATAL('please define the coordinate_convert_func funtion in %s plugin' % (self.__class__.__name__))
            box = coordinate_convert_func(pred[..., :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (pred[..., 5:] > conf_thres).nonzero(as_tuple=False).T
                pred = torch.cat(
                    (box[i], pred[i, j + 5, None], j[..., None].float()), 1)
            else:  # best class only
                conf, j = pred[..., 5:].max(1, keepdim=True)
                pred = torch.cat((box, conf, j.float()), 1)[
                    conf.view(-1) > conf_thres]
            # Check shape
            n = pred.shape[0]  # number of boxes
            if not n:
                return [], [], []
            elif n > max_nms:
                # sort by confidence
                pred = pred[pred[..., 4].argsort(descending=True)[:max_nms]]

            c = pred[..., 5:6] * (0 if agnostic else max_wh)
            boxes, scores = pred[..., :4] + c, pred[..., 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]

            output = pred[i].cpu().numpy()
            return output[..., 5], output[..., :4] / 640, output[..., 4]


@register_plugin(PluginType.Metric, '1.0')
class YOLOV5tflitemAPMetric(YOLOV5onnxmAPMetric):
    def __call__(self, pred, target):
        batch = pred[0].shape[0]

        def xywh2yxyx(x):
            # Convert boxes from [x, y, w, h] to [xmin, ymin, xmax, ymax]
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[..., 1] = x[..., 0] - x[..., 2] / 2
            y[..., 0] = x[..., 1] - x[..., 3] / 2
            y[..., 3] = x[..., 0] + x[..., 2] / 2
            y[..., 2] = x[..., 1] + x[..., 3] / 2
            return y

        for i in range(batch):
            label_id_list, boxes_list, score_list = self.decode_output(
                pred[i], coordinate_convert_func=xywh2yxyx)
            boxes_list = boxes_list * 640
            self.combine_predict_label_and_extract_obj_all_class(i, label_id_list, boxes_list, score_list, target)


@register_plugin(PluginType.Metric, '1.0')
class YOLOV5cocomAPMetric(YOLOV5onnxmAPMetric):
    """
    This YOLOV5cocomAPMetric is used for the metric of yolov5_onnx model in Optimizer.

    The input image size of facebox model is 300x300.
    score_threshold=0.25, iou_threshold=0.45
    """

    def __call__(self, pred, target):
        batch = pred[0].shape[0]

        def xywh2xyxy(x):
            # Convert nx4 boxes from [x, y, w, h] to [xmin, ymin, xmax, ymax]
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[..., 0] = x[..., 0] - x[..., 2] / 2
            y[..., 1] = x[..., 1] - x[..., 3] / 2
            y[..., 2] = x[..., 0] + x[..., 2] / 2
            y[..., 3] = x[..., 1] + x[..., 3] / 2
            return y
        for i in range(batch):
            label_id_list, boxes_list, score_list = self.decode_output(
                pred[i], coordinate_convert_func=xywh2xyxy)
            self.combine_predict_label_and_extract_obj_all_class(i, label_id_list, boxes_list, score_list, target)


@register_plugin(PluginType.Metric, '1.0')
class YOLOXcocomAPMetric(YOLOV5onnxmAPMetric):
    def __call__(self, pred, target):
        def meshgrid(*tensors):
            _TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]
            if _TORCH_VER >= [1, 10]:
                return torch.meshgrid(*tensors, indexing="ij")
            else:
                return torch.meshgrid(*tensors)

        def _decode(outputs):
            grids = []
            strides = []
            hw = [[80, 80], [40, 40], [20, 20]]
            strides_list = [8, 16, 32]
            for (hsize, wsize), stride in zip(hw, strides_list):
                yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
                grid = torch.stack((xv, yv), 2).view(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                strides.append(torch.full((*shape, 1), stride))

            grids = torch.cat(grids, dim=1).type(torch.FloatTensor)
            strides = torch.cat(strides, dim=1).type(torch.FloatTensor)

            outputs = torch.cat([
                (outputs[..., 0:2] + grids) * strides,
                torch.exp(outputs[..., 2:4]) * strides,
                outputs[..., 4:]
            ], dim=-1)
            return outputs

        def xywh2xyxy(x):
            # Convert nx4 boxes from [x, y, w, h] to [xmin, ymin, xmax, ymax]
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[..., 1] = x[..., 0] - x[..., 2] / 2
            y[..., 0] = x[..., 1] - x[..., 3] / 2
            y[..., 3] = x[..., 0] + x[..., 2] / 2
            y[..., 2] = x[..., 1] + x[..., 3] / 2
            return y
        batch = pred[0].shape[0]
        for i in range(batch):
            _pred = _decode(pred[0][i:i+1])
            label_id_list, boxes_list, score_list = self.decode_output(
                _pred, coordinate_convert_func=xywh2xyxy)
            self.combine_predict_label_and_extract_obj_all_class(i, label_id_list, boxes_list, score_list, target)


@register_plugin(PluginType.Metric, '1.0')
class YOLOV4OnnxmAPMetric(YOLOVOCmAPMetric):
    """
    This YOLOV4OnnxmAPMetric is used for the metric of yolov4_onnx model in Optimizer.

    The input image size of facebox model is 416x416.
    score_threshold=0.25, iou_threshold=0.45
    """

    def __init__(self, input_size=416, score_threshold=0.25, iou_threshold=0.45, layout='nhwc'):
        super().__init__()
        self.anchors = np.array([[[12., 16.], [19., 36.], [40., 28.]],
                                 [[36., 75.], [76., 55.], [72., 146.]],
                                 [[142., 110.], [192., 243.], [459., 401.]]]
                                )
        self.strides = [8, 16, 32]
        self.input_size = float(input_size)  # 416
        self.score_threshold = float(score_threshold)  # 0.25
        self.iou_threshold = float(iou_threshold)  # 0.45
        self.method = 'nms'
        self.sigma = 0.3
        self.layout = layout

    def __call__(self, pred, target):
        """
        #pred:[concat0(batch,52,52,3,85), concat1(batch,26,26,3,85), concat2(batch,13,13,3,85)]
        #target:[labels_index, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, [height, width]]
        """
        assert len(pred) == 3, OPT_FATAL(
            'please check the outputs number(should be 3) in Yolov4_onnx model')
        try:
            _pred = []
            featuremap_size_list = [52, 26, 13]
            for idx, pd in enumerate(sorted(pred, key=lambda x: x.size(), reverse=True)):
                if self.layout == 'nchw':
                    if pd.dim == 4:
                        pd = pd.transpose(0, 2, 3, 1)
                _pred.append(pd.reshape(-1, featuremap_size_list[idx], featuremap_size_list[idx], 3, 85))
            pred = _pred
        except:
            OPT_FATAL('output tensor can not be reshape into certain shape')
        batch = pred[0].shape[0]
        output_num = len(pred)

        pred_list = []
        org_img_h = 5000
        org_img_w = 5000
        pred_bbox_list = []
        for pd in pred:
            pred_list.append(pd.cpu().numpy())
            org_img_h = min(org_img_h, pd.shape[1])
            org_img_w = min(org_img_w, pd.shape[2])
        org_img_shape = (org_img_h*32, org_img_w*32)

        for b in range(batch):
            pred_single = []
            for num in range(output_num):
                pred_single.append(pred_list[num][b:b+1])
            pred_bbox = self.postprocess_bbbox(pred_single)
            pred_bbox = self.filter_boxes(pred_bbox, org_img_shape)
            pred_bbox = self.nms(pred_bbox)
            pred_bbox_list.append(pred_bbox)

        self.process_predict(batch, pred_bbox_list, target)

    def process_predict(self, batch, pred_bbox_list, target):
        for b in range(batch):
            predict = defaultdict()
            label_id_list = []
            boxes_list = []
            score_list = []
            pred_bbox = pred_bbox_list[b]
            for j, bbox in enumerate(pred_bbox):
                # xmin, ymin, xmax, ymax => ymin, xmin, ymax, xmax
                boxes_list.append([bbox[1]/self.input_size, bbox[0]/self.input_size,
                                   bbox[3]/self.input_size, bbox[2]/self.input_size])
                score_list.append(bbox[4])
                label_id_list.append(int(bbox[5]))
            self.combine_predict_label_and_extract_obj_all_class(b, label_id_list, boxes_list, score_list, target)

    def reset(self):
        super().reset()

    def compute(self):

        self.eval_mAP(self.predicts, self.targets, 90, 1)
        return self.mAP

    def postprocess_bbbox(self, pred_bbox, XYSCALE=[1, 1, 1]):
        for i, pred in enumerate(pred_bbox):
            conv_shape = pred.shape
            output_size = conv_shape[1]
            conv_raw_dxdy = pred[..., 0:2]
            conv_raw_dwdh = pred[..., 2:4]
            xy_grid = np.meshgrid(np.arange(output_size),
                                  np.arange(output_size))
            xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

            xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
            xy_grid = xy_grid.astype(float)

            pred_xy = ((special.expit(conv_raw_dxdy) *
                        XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * self.strides[i]
            pred_wh = (np.exp(conv_raw_dwdh) * self.anchors[i])
            pred[..., 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

        pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1]))
                     for x in pred_bbox]  # [[1,13,13,3,85]] -->[1*13*13*3,85]
        pred_bbox = np.concatenate(pred_bbox, axis=0)
        return pred_bbox

    def filter_boxes(self, pred_bbox, org_img_shape):
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[..., 0:4]
        pred_conf = pred_bbox[..., 4]
        pred_prob = pred_bbox[..., 5:]

        # box shape (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[..., :2] - pred_xywh[..., 2:] * 0.5,
                                    pred_xywh[..., :2] + pred_xywh[..., 2:] * 0.5], axis=-1)
        org_h, org_w = org_img_shape
        resize_ratio = min(self.input_size / org_w, self.input_size / org_h)

        dw = (self.input_size - resize_ratio * org_w) / 2
        dh = (self.input_size - resize_ratio * org_h) / 2

        pred_coor[..., 0::2] = 1.0 * (pred_coor[..., 0::2] - dw) / resize_ratio
        pred_coor[..., 1::2] = 1.0 * (pred_coor[..., 1::2] - dh) / resize_ratio

        # handle boxes that exceed boundaries
        pred_coor = np.concatenate([np.maximum(pred_coor[..., :2], [0, 0]),
                                    np.minimum(pred_coor[..., 2:], [org_w - 1, org_h - 1])], axis=-1)
        invalid_mask = np.logical_or(
            (pred_coor[..., 0] > pred_coor[..., 2]), (pred_coor[..., 1] > pred_coor[..., 3]))
        pred_coor[invalid_mask] = 0

        # filter some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(
            pred_coor[..., 2:4] - pred_coor[..., 0:2], axis=-1))
        valid_mask = np.logical_and(
            (valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # filter some boxes with higher scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.score_threshold
        mask = np.logical_and(valid_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate([coors, scores[..., np.newaxis], classes[..., np.newaxis]], axis=-1)

    def nms(self, bboxes):
        """
        param bboxes: (xmin, ymin, xmax, ymax, score, class)
        """
        classes_in_img = list(set(bboxes[..., 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[..., 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[..., 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate(
                    [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = BasemAPMetric.cal_iou_yxyx(
                    best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)
                assert self.method in ['nms', 'soft-nms']
                if self.method == 'nms':
                    iou_mask = iou > self.iou_threshold
                    weight[iou_mask] = 0.0

                if self.method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / self.sigma))

                cls_bboxes[:, 4] = cls_bboxes[..., 4] * weight
                score_mask = cls_bboxes[..., 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes


@register_plugin(PluginType.Metric, '1.0')
class YOLOV4tflitemAPMetric(YOLOV4OnnxmAPMetric):
    """
    This YOLOV4tflitemAPMetric is used for the metric of yolov4_tflite model in Optimizer.
    """

    def __init__(self, input_size=512, score_threshold=0.25, iou_threshold=0.45):
        super().__init__(input_size, score_threshold, iou_threshold)

    def __call__(self, pred, target):
        """
        #pred:[concat0(batch,16278,4), concat1(batch,16278,80)]
        #target:[labels_index, bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax]
        """
        assert len(pred) == 2, OPT_FATAL(
            'please check the outputs number(should be 2) in Yolov4_tflite model')
        batch = pred[0].shape[0]
        box_num = pred[0].shape[1]

        pred_bbox_list = []
        targets_list = []
        for b in range(batch):
            targets = {}
            for k, v in target.items():
                targets.update({k: v[b].numpy()})
            targets_list.append(targets)
            pred_box = pred[0][b].cpu().numpy().reshape(-1, 4)
            score = np.ones((box_num, 1))
            class_score = pred[1][b].cpu().numpy().reshape(-1, 80)
            pred_bbox = np.concatenate((pred_box, score, class_score), axis=1)
            org_img_shape = [self.input_size, self.input_size]
            pred_bbox = self.filter_boxes(pred_bbox, org_img_shape)
            pred_bbox = self.nms(pred_bbox)
            pred_bbox_list.append(pred_bbox)

        self.process_predict(batch, pred_bbox_list, targets_list)

    def reset(self):
        super().reset()

    def compute(self):
        self.eval_mAP(self.predicts, self.targets, 90, 1)
        return self.mAP

    def report(self):
        return "Yolov4 tflite mAP accuracy is %f" % (self.compute())


@register_plugin(PluginType.Metric, '1.0')
class YOLOV3TfmAPMetric(YOLOVOCmAPMetric):
    """
    This YOLOV3TfmAPMetric is used for the metric of yolov3_tf model in Optimizer.

    The input image size of cocodataset is 416x416.
    score_threshold=0.6, iou_threshold=0.5
    """

    def __init__(self, image_shape=416, score_threshold=0.6, iou_threshold=0.5, num_class=80):
        super().__init__()
        self.anchors = np.array([[[116., 90.], [156., 198.], [373., 326.]],
                                 [[30., 61.], [62., 45.], [59., 119.]],
                                 [[10., 13.], [16., 30.], [33., 23.]]])
        self.image_shape = float(image_shape)  # 416
        self.score_threshold = float(score_threshold)  # 0.6
        self.iou_threshold = float(iou_threshold)  # 0.5
        self.num_class = int(num_class)  # 80

    def __call__(self, pred, target):
        """
        #pred:[concat0(batch,13,13,255), concat1(batch,26,26,255), concat2(batch,52,52,255)]
        #target:[labels_index, bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax]
        """
        assert len(pred) == 3, OPT_FATAL(
            'please check the outputs number(should be 3) in Yolov3_tf model')
        batch = pred[0].shape[0]

        pred_list = []
        for pd in pred:
            pred_list.append(pd.cpu().numpy())

        for i in range(batch):
            boxes_, scores_, classes_ = self.yolo_eval(pred)
            self.combine_predict_label_and_extract_obj_all_class(i, classes_, boxes_, scores_, target)

    def reset(self):
        super().reset()

    def compute(self):

        self.eval_mAP(self.predicts, self.targets, 90, 1)
        return self.mAP

    def report(self):
        return "Yolov3 tf mAP accuracy is %f" % (self.compute())

    def yolo_eval(self, conv_outputs):
        num_layers = len(conv_outputs)
        input_shape = np.shape(conv_outputs[0])[1]*32
        boxes = []
        box_scores = []
        for l in range(num_layers):
            anchors = self.anchors[l]
            _boxes, _box_scores = self.yolo_boxes_and_scores(conv_outputs[l], anchors, input_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = torch.Tensor(np.concatenate(boxes)).type(torch.FloatTensor)
        box_scores = torch.Tensor(np.concatenate(box_scores)).type(torch.FloatTensor)

        mask = box_scores >= self.score_threshold
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(self.num_class):
            class_boxes = boxes[mask[:, c]]
            class_box_scores = box_scores[:, c][mask[:, c]]
            nms_idx = torchvision.ops.nms(class_boxes, class_box_scores, self.iou_threshold)
            class_boxes = torch.index_select(class_boxes, 0, nms_idx)
            class_box_scores = torch.index_select(class_box_scores, 0, nms_idx)
            classes = torch.ones_like(class_box_scores, dtype=torch.int32) * c
            boxes_.append(class_boxes.numpy())
            scores_.append(class_box_scores.numpy())
            classes_.append(classes.numpy())
        boxes_ = np.concatenate(boxes_)
        scores_ = np.concatenate(scores_)
        classes_ = np.concatenate(classes_)

        return boxes_, scores_, classes_

    def yolo_boxes_and_scores(self, conv_output, anchors, input_shape):
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(conv_output, anchors, input_shape)
        boxes = self.yolo_correct_boxes(box_xy, box_wh, input_shape)
        boxes = np.reshape(boxes, [-1, 4])
        box_scores = np.reshape(box_confidence * box_class_probs, [-1, self.num_class])
        return boxes, box_scores

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        new_shape = np.round(self.image_shape * input_shape/self.image_shape)
        offset = (input_shape-new_shape)/2./input_shape
        scale = input_shape/new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)
        return boxes

    def yolo_head(self, conv_output, anchors, input_shape):
        num_anchors = len(anchors)
        conv_output = conv_output.data.cpu().numpy()
        # Reshape to batch, height, width, num_anchors, box_params
        anchors_tensor = np.reshape(anchors, [1, 1, 1, num_anchors, 2])

        grid_shape = conv_output.shape[1]
        grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape), [-1, 1, 1, 1]),
                         [1, grid_shape, 1, 1])
        grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape), [1, -1, 1, 1]),
                         [grid_shape, 1, 1, 1])
        grid = np.concatenate([grid_x, grid_y], axis=-1)
        grid = grid.astype(np.float32)

        # convert [1,13,13,255] -> [1,13,13,3,85]
        conv_output = np.reshape(
            conv_output, [-1, grid_shape, grid_shape, num_anchors, self.num_class + 5])

        # normalize box and sigmoid confidence and probs
        box_xy = (special.expit(conv_output[..., :2]) + grid) / float(grid_shape)
        box_wh = np.exp(conv_output[..., 2:4]) * anchors_tensor / float(input_shape)
        box_confidence = special.expit(conv_output[..., 4:5])
        box_class_probs = special.expit(conv_output[..., 5:])

        return box_xy, box_wh, box_confidence, box_class_probs
