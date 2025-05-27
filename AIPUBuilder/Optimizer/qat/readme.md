Compass_QAT support finetuning models by using quantization aware training with respect to Zhouyi NPU. This document mainly introduces its basic functions and usage.

# 1. Functions

Compass_QAT workflows:

- prepare(), set the corresponding parameters according to the configuration file
- fuse(), the corresponding modules are fused according to the Compass operator specification. The fusion requires writing the fusion method of the corresponding module
- evaluate_loop()， evaluates the accuracy of the model
- finetune(), according to the configured dataset, train_plugin in the form of plugin is used to train the model
- export(), save the trained model

# 2. Resnet_V1_50 demo

(To run the following demo, you need to install the python package of 'AIPUBuilder' firstly.)

## 2.1 model download

```python
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet_model = models.resnet50(pretrained=True).to(device)
    torch.save(resnet_model, 'resnet50_pretrained.pt')
```

## 2.2 configuration

resnet_v1_50.cfg includes：

```python
[Common]
input_model = resnet50_pretrained.pt
model_name = resnet50

weight_bits = 8
activation_bits = 8

calibration_strategy_for_activation = extrema
calibration_strategy_for_weight = extrema
quantize_method_for_weight = per_tensor_symmetric_restricted_range
quantize_method_for_activation = per_tensor_symmetric_full_range

train_data = ./model_dataset/resnet50/dataset/dataset_20.npy
train_label = ./model_dataset/resnet50/dataset/label_20.npy
data = ./model_dataset/resnet50/dataset/dataset_1000.npy
label = ./model_dataset/resnet50/dataset/label_1000.npy

dataset = numpynhwc2nchwdataset
metric = topkmetric
train = ResNet50TrainLoop

train_batch_size = 50
metric_batch_size = 100
train_shuffle = True
input_shape = [[1, 3, 224, 224]]

output_dir = ./

```

- input_model: Specify the path to the torch model
- train_data/train_label: Specifies the training dataset path used for finetune
- train: Configure the finetune plugin
- train_batch_size: batch size for training
- data/label/dataset/metric/metric_batch_size/calibration_strategy_for_activation/calibration_strategy_for_weight/quantize_method_for_weight/quantize_method_for_activation/weight_bits/activation_bits: The configuration method of these fields are exactly the same as the Compass_Optimizer.

to use the above configurations：

```python
python3 qatmain.py -c resnet_v1_50.cfg
```

logs will be like:

```python
$ p3 qatmain.py -c src/test/resnet_50.cfg
[I] [OPT] [11:19:52]: [arg_parser] is running.
[I] [QAT] [11:19:52]: [prepare] is running...
[I] [QAT] [11:19:52]: [fuse] the module is running...
[I] [QAT] [11:19:53]: cosine distance of 0 output: 0.9999999999998461
[I] [QAT] [11:19:53]: [evaluate] the fused_module is running...
evaluate the model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:25<00:00,  2.57s/it]
[I] [QAT] [11:20:18]: 	top-1 accuracy is 0.722000
[I] [QAT] [11:20:18]: [finetune] the fused_module is running...
[I] [QAT] [11:20:18]:
now train the epoch = 1/3
[I] [QAT] [11:20:19]: loss = 1.166358232498169, [0.05]
evaluate the model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:26<00:00,  2.62s/it]
[I] [QAT] [11:20:46]: 		 now running evaluate function(10):top-1 accuracy is 0.731000
[I] [QAT] [11:20:46]:
now train the epoch = 2/3
[I] [QAT] [11:20:46]: loss = 0.6217561960220337, [0.05]
evaluate the model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:26<00:00,  2.61s/it]
[I] [QAT] [11:21:12]: 		 now running evaluate function(10):top-1 accuracy is 0.733000
[I] [QAT] [11:21:12]:
now train the epoch = 3/3
[I] [QAT] [11:21:13]: loss = 0.32806044816970825, [0.05]
evaluate the model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:26<00:00,  2.63s/it]
[I] [QAT] [11:21:39]: 		 now running evaluate function(10):top-1 accuracy is 0.732000
[I] [QAT] [11:21:39]: after finetune, running evaluate function:
evaluate the model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:25<00:00,  2.54s/it]
[I] [QAT] [11:22:05]: 	top-1 accuracy is 0.732000
[I] [QAT] [11:22:05]: [export] the fused_module is running...
[I] [QAT] [11:22:05]: serialize the float IR:
[I] [QAT] [11:22:06]: serialize the qat IR:
```

For convenience, use the following script to illustrate the QAT process again:

```python
qat_handler = AIPUQATMaster(config)
quantizer = qat_handler.qat_quantizer

qat_handler.prepare()
QAT_INFO(f"orignal resnet 50 evaluated acc:")
qat_handler.evaluate_loop(quantizer.model)

QAT_INFO(f"begin to fuse the resnet50 and transform to zhouyi qnn:")
fused_model = quantizer.fuse()
QAT_INFO(f"fused resnet 50 evaluated acc:")
qat_handler.reset_metric()
qat_handler.evaluate_loop(fused_model)

QAT_INFO(f"fused resnet 50 qat_train loop:")
qat_handler.finetune(qat_handler.qat_quantizer.fused_module)
qat_handler.export()

```


## 2.3 Qoperator

Qoperator represents the module implementation of Compass operator. For example, for the convolution module, the module that implements QConvolution under Compass_QAT is used to simulate the calculation process of convolution in the original model.

Compass_QAT use *register_operator*  to register a Qoperator. The Qconvolution requires the implementation of forward() and serialize() functions, where the forward() method completes the forward computation and serialize() method calls `AIPUBuilder.ops` method to implement the serialization function of the operator. Please refer to `AIPUBuilder/Optimizer/qat/src/ops/`

## 2.4 fuse pattern

use *register_fusion_pattern* to register the fuse pattern that a Compass operator needed, for example:

```python
@register_fusion_pattern((nn.ReLU6, (nn.BatchNorm2d, nn.Conv2d)))  # 1
@register_fusion_pattern((nn.ReLU, (nn.BatchNorm2d, nn.Conv2d)))  # 0, the highest priority
class ConvBNActFusion:
    def __init__(self, quantizer, node):
        pass

    def fuse(self, graph_module, modules):
        fused_graph = graph_module.graph
        act_function = self.act_function
        if self.bn_node is not None:
            fused_conv = fuse_conv_bn_eval(self.conv, self.bn)
        else:
            fused_conv = copy.deepcopy(self.conv)

        hyper_params = extract_conv_hyperparams(fused_conv)
        hyper_params['act_function'] = copy.deepcopy(act_function)

        q_conv = QConvolution2D(self.conv_name, **hyper_params, conv_node=fused_conv)
        q_conv.weight.data = fused_conv.weight
        if q_conv.bias is not None:
            q_conv.bias.data = fused_conv.bias

        replace_node_module(self.conv_node, modules, q_conv)

        if self.bn_node is not None:
            replace_node_module(self.bn_node, modules, torch.nn.Identity())
            self.bn_node.replace_all_uses_with(self.conv_node)
            fused_graph.erase_node(self.bn_node)

        if self.act_node is not None:
            replace_node_module(self.act_node, modules, torch.nn.Identity())
            self.act_node.replace_all_uses_with(self.conv_node)
            fused_graph.erase_node(self.act_node)

```

In the fuse() stage, the original convolution (or multiple modules that need to be fused) will be changed to the QConvolution module introduced above.

Therefore, if there are modules that Compass fuser and Qoperator not supported, you need to manually add the fuser and Qoperator for them.

## 2.5 finetune

In order to make it easier for users to customize their own finetune process, Compass_QAT uses train_plugin to access the user's finetune method, use *register_plugin(PluginType.Train, ‘1.0’)* to register, see `AIPUBuilder/Optimizer/qat/src/plugin/aipubt_train_resnet50.py` for example.

## 2.6 evaluate_loop

This method is mainly used to evaluate the accuracy of the model. For example, the model accuracy before fusion and the model accuracy after finetune.

## 2.6 export

This method will try to save the finetuned model as Compass IR format firstly, and if fails then try to export it as onnx format.
