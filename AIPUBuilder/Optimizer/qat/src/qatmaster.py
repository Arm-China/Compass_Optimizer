# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


import sys
import torch
from torch.utils.data import DataLoader

from AIPUBuilder.Optimizer.framework.opt_register import *
from .plugin import *
from .config import QATConfig
from .qatlogger import QAT_ERROR, QAT_WARN, QAT_INFO

from AIPUBuilder.Optimizer.framework.opt_register import (
    TRAIN_PLUGIN_DICT,
    QUANTIZE_DATASET_DICT,
    QUANTIZE_METRIC_DICT,
)
from AIPUBuilder.Optimizer.config import MetricField
from AIPUBuilder.Optimizer.logger import tqdm
from .quantizer import PytorchQuantizer

from .qatfield import TrainField
from .qatregister import QAT_COMPASS_OPERATORS
from .qinfo import QuantStage


class AIPUQATMaster(object):
    def __init__(self, config):
        self.qat_config = QATConfig(config)
        self.qat_quantizer = self.get_quantizer(self.qat_config)

        self.train_dataloader = None
        self.train_loops = []

        self.evaluate_dataloader = None
        self.metrics = []

        self.device = self.qat_config.get("device")

    def fuse(self):
        return self.qat_quantizer.fuse()

    def prepare(self):
        workers = self.qat_config.get("dataloader_workers", 1)
        dataset_field = self.qat_config.get("dataset").lower()
        train_plugin = self.qat_config.get("train")
        if train_plugin != "":
            train_data_path = self.qat_config.get("train_data")
            train_label_path = self.qat_config.get("train_label")
            if dataset_field != "" and dataset_field in QUANTIZE_DATASET_DICT:
                train_batch_size = self.qat_config.get("train_batch_size")
                dataset = QUANTIZE_DATASET_DICT[dataset_field](
                    train_data_path, train_label_path
                )
                collate_fn = (
                    dataset.collate_fn if hasattr(dataset, "collate_fn") else None
                )
                self.train_dataloader = DataLoader(
                    dataset,
                    batch_size=train_batch_size,
                    shuffle=self.qat_config.get("train_shuffle"),
                    num_workers=workers,
                    collate_fn=collate_fn,
                )

                fn_arg_dict = TrainField.get_train(train_plugin)
                for fn, argl in fn_arg_dict.items():
                    for arg in argl:
                        self.train_loops.append(TRAIN_PLUGIN_DICT[fn.lower()](*arg))
                self.qat_config.set('train_dataloader', self.train_dataloader)
            else:
                QAT_WARN(f"Despite setted train method, the dataset is missing.")

        metric_plugin = self.qat_config.get("metric")
        if metric_plugin != "":
            eval_data_path = self.qat_config.get("data")
            eval_label_path = self.qat_config.get("label")
            if dataset_field != "" and dataset_field in QUANTIZE_DATASET_DICT:
                if eval_data_path == "" or eval_label_path == "":
                    QAT_WARN(f"dataset has setted, but data or label in cfg file is setted = ''.")
                metric_batch_size = self.qat_config.get("metric_batch_size")
                dataset = QUANTIZE_DATASET_DICT[dataset_field](
                    eval_data_path, eval_label_path
                )
                collate_fn = (
                    dataset.collate_fn if hasattr(dataset, "collate_fn") else None
                )
                self.evaluate_dataloader = DataLoader(
                    dataset,
                    batch_size=metric_batch_size,
                    shuffle=False,
                    num_workers=workers,
                    collate_fn=collate_fn,
                )

                fn_arg_dict = MetricField.get_metric(metric_plugin)
                for fn, argl in fn_arg_dict.items():
                    for arg in argl:
                        self.metrics.append(QUANTIZE_METRIC_DICT[fn.lower()](*arg))
                self.qat_config.set('evaluate_dataloader', self.evaluate_dataloader)
            else:
                QAT_WARN(f"Despite setted metric method, the dataset is missing.")

    def get_quantizer(self, config):
        input_model = config.get("input_model")
        qat_quantizer = None
        if input_model.endswith(".pt"):
            qat_quantizer = PytorchQuantizer(config)
        else:
            QAT_ERROR(f"unsupport input model type, only support .pt model")
        return qat_quantizer

    def fused_module_forward(self, input_data):
        if self.qat_quantizer is None:
            QAT_ERROR(f"fused_module_forward failed because qat_quantizer is None")
        if self.qat_quantizer.fused_module is None:
            if self.qat_quantizer.fuse() is None:
                QAT_ERROR(f"fuse process failed when wanting  fused_module_forward")

        outputs = self.qat_quantizer.forward(
            self.qat_quantizer.fused_module, input_data
        )
        return outputs

    def set_stage(self, model, stage='qat'):
        for m in model.modules():
            if isinstance(m, tuple(QAT_COMPASS_OPERATORS.keys())):
                m.quant_stage = QuantStage.str_to_quantstage(stage)

    def finetune(self, model, serialize_func=None):
        self.set_stage(model,)
        self.train_loop(model, serialize_func)
        return model

    def _parallel(self, model):
        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            model = torch.nn.DataParallel(model)
            model = model.cuda()
        return model

    def train_loop(self, model, serialize_func=None):
        if self.qat_config.get("parallel"):
            model = self._parallel(model)
        for train_loop in self.train_loops:
            train_loop(model,
                       self.train_dataloader,
                       self.evaluate_dataloader,
                       eval_func=self.evaluate_loop,
                       metric=self.metrics,
                       device=self.qat_config.get("device"),
                       serialize_func=serialize_func)

    def evaluate_loop(self, model, eval_dataloader=None, metrics=None, prefix_msg="", stage='fp32'):
        model.eval()
        self.set_stage(model, stage)
        if eval_dataloader is None:
            eval_dataloader = self.evaluate_dataloader
        if metrics is None:
            metrics = self.metrics

        for m in metrics:
            m.reset()
            # QAT_INFO(f"after reset m.correct: {m.correct}")

        if eval_dataloader is None:
            QAT_ERROR(f"if want to evaluate the model, please enable evaluate_dataloader")
            return

        with tqdm(eval_dataloader, desc="evaluate the model", file=sys.stdout) as pbar:
            with torch.no_grad():
                for _, (input_data, target) in enumerate(pbar):
                    for j, d in enumerate(input_data):
                        input_data[j] = d.to(self.device).float()
                        if isinstance(target, (list, tuple)):
                            target = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in target]
                        else:
                            target = target.to(self.device) if isinstance(target, torch.Tensor) else target
                    outs = model(*input_data)
                    for metric in metrics:
                        metric([outs], target)
        torch.cuda.empty_cache()
        for m in metrics:
            QAT_INFO(f"\t{prefix_msg}{m.report()}")

    def reset_metric(self):
        for m in self.metrics:
            m.reset()

    def export(self, model=None, prefix="", input_shapes=[]):
        torch.save(model, f'{self.qat_quantizer.output_dir}/{self.qat_quantizer.model_name}_{prefix}.pt')
        self.set_stage(model, 'fp32')
        if model is not None:
            model.eval()
        with torch.no_grad():
            try:
                self.qat_quantizer.serialize(model, prefix=prefix, input_shapes=input_shapes)
            except Exception as e:
                QAT_INFO(f"try export compass quantized IR failed, now try to export onnx model. and meets err: {e}")
                self.qat_quantizer.export(model)

    def run(self):
        QAT_INFO(f"[prepare] is running...")
        self.prepare()
        QAT_INFO(f"[fuse] the module is running...")
        fused_module = self.fuse()
        QAT_INFO(f"[evaluate] the fused_module is running...")
        self.evaluate_loop(fused_module)
        QAT_INFO(f"[finetune] the fused_module is running...")
        finetune_module = self.finetune(fused_module, serialize_func=self.export)
        QAT_INFO(f"[export] the fused_module is running...")
        self.export(finetune_module)
