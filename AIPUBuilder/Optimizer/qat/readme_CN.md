Compass_QAT支持通过模拟Compass量化调优训练QAT模型，使模型在Zhouyi NPU部署的时候达到更高的精度和更快的性能。本文档主要介绍其基本功能和使用方式。

# 1. 功能

Compass_QAT工作流程包括：

- prepare(), 根据配置文件，设置相应的参数，得到基础的torch module
- fuse(), 根据Compass算子规范对相应的module进行融合，该融合前提，需要编写相应module的融合方法。融合之后得到fused module
- evaluate_loop()， 该方法主要评估模型精度
- finetune(), 根据配置的数据集， 使用plugin形式的train_plugin对模型进行训练
- export(), 导出模型，目前是Compass Float and Quantized IR

# 2. Resnet_V1_50示例

（想要运行以下demo，需要安装`AIPUBuilder`的python包。）

## 2.1 模型下载

```python
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet_model = models.resnet50(pretrained=True).to(device)
    torch.save(resnet_model, 'resnet50_pretrained.pt')
```

## 2.2 配置文件

Compass_QAT沿用Compass NN compiler的方式，通过配置文件方式。例如resnet_v1_50.cfg如下：

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

- input_model: 指定该torch模型的路径， 使用方式与Compass NN compiler中parser配置字段相同
- train_data/train_label: 指定了finetune用的数据集，通过dataset=numpynhwc2nchwdataset plugin读取该数据集，该使用方法与Compass NN compiler的Optimizer配置字段相同。
- train: 配置finetune的plugin，具体可见后续介绍。
- train_batch_size: finetune时数据batch大小。
- data/label/dataset/metric/metric_batch_size/calibration_strategy_for_activation/calibration_strategy_for_weight/quantize_method_for_weight/quantize_method_for_activation/weight_bits/activation_bits等字段配置方式与Compass NN compiler的Optimizer配置字段完全相同。

上述配置文件的使用方式：

```python
python3 qatmain.py -c resnet_v1_50.cfg
```

运行log:

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

为方便介绍，用以下script示意QAT的过程

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

Qoperator表示Compass算子模拟qat量化反量化的module实现，例如针对convolution module，在Compass_QAT下实现QConvolution的module用于模拟原始模型中convolution的计算过程。

Compass_QAT采用 *register_operator* 的修饰器注册相应的Qoperator。该Qconvolution需要实现forward()和serialize()函数，其中forward()方法完成torch module forward的功能，serialize()方法调用`AIPUBuilder.ops`的方法实现算子的序列化功能。详见`AIPUBuilder/Optimizer/qat/src/ops/`

## 2.4 fuse pattern

使用 *register_fusion_pattern* 注册需要fuse的pattern， 例如以conv为例：

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

在fuse()阶段，会将原始convolution(或者多个需要融合的module)变为上面介绍的QConvolution module.

在fuse之后，原始torch module会转化为为Compass_QAT定义的Qoperator算子。

因此，如果原始模型中有Compass fuser和Qoperator没有支持的module, 需要手动添加该module的fuser和Qoperator

## 2.5 finetune

为了更方便用户定制自己的finetune过程，Compass_QAT使用train_plugin的方式接入用户的finetune方法，使用 *register_plugin(PluginType.Train, ‘1.0’)* 的方式注册。例如`AIPUBuilder/Optimizer/qat/src/plugin/aipubt_train_resnet50.py`

## 2.6 evaluate_loop

该方法主要用于评估模型的精度，例如在fuse之前的模型精度，finetune之后的模型精度。

## 2.7 export

该方法用于保存训练后的模型，会优先保存为Compass IR格式，如失败则会转为导出onnx格式
