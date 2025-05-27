# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch

from AIPUBuilder.Optimizer.framework import register_plugin, PluginType
from ..qatlogger import QAT_INFO, QAT_ERROR, QAT_DEBUG
from ..qatregister import QATBaseTrainLoop
from ..config import default_device


@register_plugin(PluginType.Train, '1.0')
class ResNet50TrainLoop(QATBaseTrainLoop):
    def __init__(self,
                 learning_rate=0.00001,
                 epochs=1,
                 **kwargs):
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.default_device = default_device()

    def get_loss(self, pred, target):
        outs = torch.nn.functional.log_softmax(pred, dim=1)
        loss = torch.nn.functional.nll_loss(outs, target)
        return loss

    def get_opt(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.learning_rate)

    def __call__(self, model, train_dataloader, evaluate_dataloader=None, eval_func=None, metric=None, device=None, serialize_func=None):
        size = len(train_dataloader.dataset)
        batch_size = train_dataloader.batch_size
        opt = self.get_opt(model)

        def save_model(model, epoch: int):
            model.eval()
            serialize_func(model, prefix=f'round{epoch}')

        max_round = 0
        max_acc = 0.0
        for epoch in range(self.epochs):
            save_model(model, epoch)
            cur_input = None
            log_loss_intervals = []
            log_loss_num = 5
            for k in range(log_loss_num):
                log_loss_intervals.append([k/log_loss_num, 1])
            for b, (input_data, target) in enumerate(train_dataloader):
                for j, d in enumerate(input_data):
                    input_data[j] = d.to(device).float()
                    if isinstance(target, (list, tuple)):
                        target = [t.to(device) if isinstance(t, torch.Tensor) else t for t in target]
                    else:
                        target = target.to(device) if isinstance(target, torch.Tensor) else target
                self.set_stage(model, 'qat')
                model.train()
                cur_input = input_data[0]
                pred = model(input_data[0])
                loss = self.get_loss(pred, target)

                # backpropagation
                loss.backward()
                opt.step()
                opt.zero_grad()

                loss = loss.item()
                current = b * batch_size + len(input_data)
                percent = current / size
                loss_msg = f"epoch={epoch}/{self.epochs}, loss = {loss}, [{percent}epoch], learning_rate={self.learning_rate}"
                QAT_DEBUG(loss_msg)

                for ik, (interval, icnt) in enumerate(log_loss_intervals):
                    if icnt > 0 and percent >= interval:
                        QAT_INFO(loss_msg)
                        log_loss_intervals[ik][1] -= 1
            model.eval()
            # update statistic after backward
            _ = model(cur_input)
            eval_func(model, evaluate_dataloader, metrics=metric, stage='fp32',
                      prefix_msg=f"epoch={epoch+1}/{self.epochs} sample_num = {current}/{size} \t now running fp32 evaluate function({len(evaluate_dataloader)}):")
            facc = metric[0].compute()
            eval_func(model, evaluate_dataloader, metrics=metric, stage='infer',
                      prefix_msg=f"epoch={epoch+1}/{self.epochs} sample_num = {current}/{size} max_acc={max_acc} max_round={max_round} \t now running qat evaluate function({len(evaluate_dataloader)}):")
            acc = metric[0].compute()
            if acc > max_acc:
                max_acc = acc
                max_round = epoch+1

        return model
