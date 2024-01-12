[TOC]

## Compass Optimizer
Compass Optimizer, 简称**OPT**, 是周易 Compass Neural Network Compiler (Python包名为AIPUBuilder) 工具链的一部分，主要负责将[Compass Unified Parser](https://github.com/Arm-China/Compass_Unified_Parser)转换后的浮点模型中间表示（IR），进一步优化（通过量化，计算图优化等手段）为适合在周易NPU硬件平台上执行的量化模型或混合精度模型中间表示。关于Compass IR规范及Compass SDK的更多介绍，参见https://aijishu.com/a/1060000000215443。

除本概要说明以外，还可以参阅`tutorial.pdf`的详细介绍。

### 主要特性
OPT的主要功能特性如下：
- 支持多种模型量化方法：逐张量量化、逐通道量化，非对称量化、对称量化
- 支持混合精度量化：如8比特、16比特混合量化，部分层量化、部分层浮点运行，自动搜索量化精度
- 支持逐层配置量化相关参数：可通过json配置文件逐层配置参数
- 支持多种常用量化校准方案：

  - Averaging: 不同batch校准数据的统计值再加权平均

  - Mean +/- N*Std: 用张量的均值方差定义量化范围，其中N为可配置参数

  - ACIQ: 参见[Post-training 4-bit quantization of convolution networks for rapid-deployment](https://arxiv.org/abs/1810.05723)

  - KLD: 参见[8-bit inference with TensorRT](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)

  - EasyQuant: 参见[EasyQuant: Post-training Quantization via Scale Optimization](https://arxiv.org/abs/2006.16669)

  - Adaround: 参见[Up or Down? Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568)

  - Adaquant: 参见[Improving Post Training Neural Quantization: Layer-wise Calibration and Integer Programming](https://arxiv.org/abs/2006.10518)

- 适配周易全系列硬件平台，针对性改进各算子量化实现及优化调整计算图结构


### 快速入门

#### 安装向导
你可以通过[Compass_Integration](https://github.com/Arm-China/Compass_Integration)中的指引来编译一个包含OPT的AIPUBuilder, 关于AIPUBuilder的使用说明，请参考[MiniPkg](https://aijishu.com/a/1060000000215443)里面的说明书。

此外，OPT可以单独运行。只要满足如下的依赖，就可以直接执行`AIPUBuilder/Optimizer/tools/optimizer_main.py`文件来运行OPT.
#### 安装依赖
* Python3 >= 3.8.5
* NumPy >= 1.22.3
* NetworkX >= 2.8
* torch >= 1.11.1
* torchvision >= 0.12.0
#### 运行OPT
OPT是以配置文件为输入来驱动的，你可以按如下示例来运行OPT：
```bash
export PYTHONPATH=./:$PYTHONPATH
python3 ./AIPUBuilder/Optimizer/tools/optimizer_main.py --cfg ./opt.cfg
```
#### 配置文件格式
所有的选项必须在 `Common` 字段里面：
- `graph` 输入Float IR的定义文件路径

- `bin` 输入Float IR的权重文件路径

- `model_name` 此模型的名称

- `dataset` 用于读取此模型对应的数据集的插件类名称（可单独用`optimizer_main.py --plugin`命令查看所有已经实现的数据集插件类）

- `calibration_data` 用于校准量化的数据集文件路径

- `calibration_batch_size` 用于校准量化的`batch_size`

- `metric` 用于计算此模型效果指标的插件类名称（可单独用`optimizer_main.py --plugin`命令查看所有已经实现的指标插件类），若不需要计算效果指标则不必设置此项

- `data` 若设置了`metric`则指定对应用到的验证数据集文件路径

- `label` 若设置了`metric`则指定对应用到的验证数据集的标注文件路径

- `metric_batch_size` 若设置了`metric`则指定对应前向运算时的`batch_size`

- `quantize_method_for_weight` 量化模型权重的方法，包括：

  - `per_tensor_symmetric_restricted_range`

  - `per_tensor_symmetric_full_range`

  - `per_channel_symmetric_restricted_range`

  - `per_channel_symmetric_full_range`

  默认为`per_tensor_symmetric_restricted_range`

- `quantize_method_for_activation` 量化模型激活响应的方法，包括：

  - `per_tensor_symmetric_restricted_range`

  - `per_tensor_symmetric_full_range`

  - `per_tensor_asymmetric`

  默认为`per_tensor_symmetric_full_range`

- `weight_bits` 量化模型权重参数的比特位宽，默认为8

- `bias_bits` 量化模型偏置参数的比特位宽，默认为32

- `activation_bits` 量化模型激活响应的比特位宽，默认为8

- `lut_items_in_bits` 量化计算部分数学函数（如sigmoid, tanh等）时用到的lut查找表大小（用2的`lut_items_in_bits`次方表示），默认为8即表项数为256项。当`activation_bits`改变时需对应调整此设置以平衡性能和精度（lut表越大精度越高，但是占用消耗的资源也越多）

- `output_dir` 输出IR和其它文件的目录

更多可配置的选项及其含义描述可执行`optimizer_main.py --field`查看。

#### 运行测试

##### 模型测试

在`AIPUBuilder/Optimizer/test/model_test/squeezenet`内给出了一个典型的模型测试用例，进入该目录执行`sh ./run.sh`即可，更多完整的模型用例可参见[Zhouyi Model Zoo](https://github.com/Arm-China/Model_zoo)。

##### 算子测试

单算子的测试可以看成是一种特殊的模型测试，按单算子的IR定义构造好测试用IR（及其输入数据）后，复用模型测试的流程即可，在`AIPUBuilder/Optimizer/test/op_test`内给出了一个典型的算子测试用例，进入该目录执行`sh ./run.sh`即可。

### OPT的处理流程和设计理念

OPT的主要处理流程如下图所示：

![opt flow](./images/opt_flow.svg)

1. 读取[Parser](https://github.com/Arm-China/Compass_Unified_Parser)产生的Float IR, 并构建生成一个内部统一图表征g。
2. 对g执行一次全零输入的前向运算以检查正确性。
3. 对g进行一轮量化前的计算图优化。
4. 基于给定的校准数据集做前向运算，并统计出图中各张量的各种统计量。
5. 对g进行一轮量化相关的计算图优化。
6. 按给定配置对相应的层做量化转换产生新的图表征qg。
7. 对qg进行量化后的计算图优化。
8. 对qg执行一次全零输入的前向运算以检查正确性。
9. 输出此优化后的量化或混合精度IR。
10. 最后根据配置决定是否导出中间张量，以及是否根据给定的验证数据集计算模型效果指标。

OPT整体采用了调度框架与具体实现相分离的机制，每个算子的实现以及模型的输入数据提供（dataset）和输出数据处理（metric）均以plugin的方式被集成到整体流程中由`OptMaster`类调度，方便使用者二次开发，扩展支持新算子或更新已有算子的实现。

### 开发指引

#### 核心数据结构
如下是OPT核心数据结构的概览：
![opt uml](./images/opt_uml.svg)

- `Dtype`定义了IR中可能出现的各种基础数据类型
- `PyTensor`为OPT中表达张量的基础类，其实际数据存储和计算通过torch.Tensor进行
- `PyNode`表征模型中层节点（layer）的概念，层与层的连接关系通过共享的边（即inputs和outputs中存储的PyTensor实例）来体现
- `PyGraph`表征整个模型结构，里面存储了所有的层节点，其拓扑结构通过内部的networkx.DiGraph实例进行维护。`QuantizeGraph`作为`PyGraph`的子类被`OptMaster`类实际使用到
- `OptMaster`类控制OPT整个的执行流程，并根据配置文件动态实例化模型的输入数据提供（dataset）类和输出数据处理（metric）类

#### 开发各类插件
最常见的OPT开发范式即为添加支持更多的Operator, Dataset, Metric插件以支持自己的专属模型，因此这里对各类插件的开发加以详细介绍。对其它OPT功能的扩充或修正（量化方式、校准算法、图优化算法等）在此从略。
##### 命名规范
建议使用以下前缀标识区分：
- aipubt_opt_op_ for the optimizer operator plugin.
- aipubt_opt_dataset_ for the dataset plugin of the optimizer.
- aipubt_opt_metric_ for the metric plugin of the optimizer.
##### 搜索路径
plugin文件的搜索顺序如下：
1. 环境变量AIPUPLUGIN_PATH所指定的路径，设定方式类似如下：
    `export AIPUPLUGIN_PATH=/home/user/aipubuilder_plugins/:$AIPUPLUGIN_PATH`
2. 当前路径下的plugin文件目录，即`./plugin/`.
##### Operator plugin编写

Operator plugin需要实现并注册两个接口：

- 使用`op_register(OpType, version)`注册forward函数`forward_function_name(self, *args)`.
- 使用`quant_register(OpType, version)`注册quantize函数`quantize_function_name(self, *args)`.

其中OpType为内置的算子类型枚举类，如果想要替换某一内置算子的实现，则注册时直接传入OpType.*layer_type_name*，并将version设定成1.0以上（内置算子版本号为1.0）。如果想要实现的是新的算子，则注册之前全局调用`register_optype('new_layer_type_name')`函数将名称注册到OpType后即可正常使用OpType.*new_layer_type_name；version表示plugin的版本号，即当有多个同名同类plugin时，会实际调用版本号更大的（同时需注意到forward函数和quantize函数是分开注册的，所以欲整体替换某一算子的实现时需保证实现的forward函数和quantize函数都具有较高的版本号）；self指向一个`PyNode`类实例（对应于IR中的某一层），其重要成员的使用方式将结合如下代码示例加以说明：

```python
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *

register_optype('DummyOP')

@op_register(OpType.DummyOP, '1024')
def dummy_forward(self, *args):
    #self.inputs and self.outputs are lists of PyTensors of this layer
    #PyTensor.betensor is the really backend tensor variable and is a instance of torch.Tensor
    inp = self.inputs[0]
    out = self.outputs[0]
    #self.constants is an ordered-dictionary for storing constant tensors, such as weights and biases
    #suggest to use self.get_constant to safely visit it
    w = self.constants['weights'] if 'weights' in self.constants else 0

    #'OPT_DEBUG, OPT_INFO, OPT_WARN, OPT_ERROR, OPT_FATAL' are basic log APIs, and only OPT_FATAL will abort execution
    OPT_INFO('layer_type=%s, layer_name=%s' % (str(self.type), self.name))

    if self.name in ['name_of_layer_x', 'name_of_layer_y'] :
        print('you can set a breakpoint here for debug usage')
    #self.attrs is an ordered-dictionary for storing the intermediate parameters, which is not writing to IR
    #suggest to use self.get_attrs to safely get a atrribute
    if self.get_attrs('layer_id') in ['2', '4', '8'] :
        print('you can also set breakpoint here in this way for debug usage')

    #self.current_batch_size indicate the current batch_size the dataloader offers
    dummy_var = inp.betensor + self.current_batch_size
    #self.quantized is flag maintained by the optimizer framework that indicates whether it's a quant_forward or normal_forward
    if self.quantized :
        #self.params is an ordered-dictionary for storing the necessary parameters
        #suggest to use self.get_param to safely get a parameter
        if self.get_param('whether_plus_one') :
            dummy_var += 1
    else :
        if self.get_param('whether_minus_one') :
            dummy_var -= 1
    out.betensor = inp.betensor if True else dummy_var

    #self.placeholders is a list where you can store temporary PyTensors for whatever you like
    if len(self.placeholders) < 1 :
        #you can use PyTensor(tensor_name) to construct an empty PyTensor,
        #or use PyTensor(tensor_name, numpy_array) to construct and initialize a PyTensor
        #dtype2nptype is a utility function in AIPUBuilder.Optimizer.utils and you can access many other utility functions here
        #Dtype defines data types NN compiler supports
        ph0 = Tensor(self.name+"/inner_temp_vars", (inp.betensor+1).cpu().numpy().astype(dtype2nptype(Dtype.FP32)))
        self.placeholders.append(ph0)
    else :
        #if the ph0 has already been put into placeholders, then we only need to update its value every time when dummy_forward is called
        self.placeholders[0].betensor = inp.betensor + 1

@quant_register(OpType.DummyOP, '1024')
def dummy_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    #PyTensor.scale is the linear quantization scale
    out.scale = inp.scale
    #PyTensor.zerop is the linear quantization zero point
    out.zerop = inp.zerop
    #PyTensor.qbits is the quantization bit width
    out.qbits = inp.qbits
    #PyTensor.dtype is the quantization Dtype information
    out.dtype = inp.dtype
    #PyTensor.qinvariant indicates whether the tensor is quantization invariant (like index values), and if it's True, the scale = 1.0, zerop=0
    out.qinvariant = inp.qinvariant
    #PyTensor.qmin and PyTensor.qmax are the clamp boundaries when tensor is quantized
    out.qmin = inp.qmin
    out.qmax = inp.qmax

    ph0 = self.placeholders[0]
    ph0.qinvariant = False
    #q_bits_weight, q_bits_bias, q_bits_activationin in self.attrs are used to carry the quantization bits information from per-layer opt_config file
    ph0.qbits = self.get_attrs('q_bits_activation')
    #q_mode_weight, q_mode_bias, q_mode_activationin in self.attrs are used to carry the quantization mode (per-tensor or per-channel, symmetric or asymmetric) information from per-layer opt_config file
    q_mode_activation = self.get_attrs('q_mode_activation')
    #get_linear_quant_params_from_tensor is a utility function in AIPUBuilder.Optimizer.utils and you can access many other utility functions here
    ph0.scale, ph0.zerop, ph0.qmin, ph0.qmax, ph0.dtype = get_linear_quant_params_from_tensor(ph0, q_mode_activation, ph0.qbits, is_signed = True)

    #you can set simple parameters to self.params which will be wrote to IR when serialize the model.
    self.params['whether_plus_one'] = True
    self.params['whether_minus_one'] = False
    #you can set complicated parameters like lookup tables to self.constants which will also be wrote to IR when serialize the model
    self.constants['lut'] = Tensor(self.name+"/lut", (torch.zeros(256)).cpu().numpy().astype(dtype2nptype(Dtype.UINT16)))

```

需要补充说明的是，optimizer初始读入float IR后会做一次normal foward以保证每个算子的quantize函数被调用之前，其forward函数一定会至少被调用过一次（forward函数被调用之前不保证一定调用过quantize函数），因此，在forward函数内正确设定的placeholder或attrs等属性值在quantize函数内可以顺利被读取，而反之则不一定可以。更详细而有实用意义的样例可以参考`AIPUBuilder/Optimizer/ops`目录下的内置算子。

##### Dataset plugin编写
Dataset plugin直接继承自`torch.utils.data.Dataset`类，实现时需提供三个公共接口`__init__, __len__ and __getitem__`，具体实例见如下NumpyDatset类：
```python
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
from torch.utils.data import Dataset
import numpy as np

@register_plugin(PluginType.Dataset, '1.0')
class NumpyDataset(Dataset):
    #when used as calibration dataset, label_file can be omitted.
    def __init__(self, data_file, label_file=None):
        self.data = None
        self.label = None
        try:
            self.data = np.load(data_file, mmap_mode='c')
        except Exception as e:
            OPT_FATAL('the data of NumpyDataset plugin should be Numpy.ndarray and allow_pickle=False.')
        if label_file is not None:
            try:
                self.label = np.load(label_file, mmap_mode='c')
            except ValueError:
                self.label = np.load(label_file, allow_pickle=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        #Assume that all preprocesses have been done before save to npy file.
        #If the graph has single input tensor,
        #the data part sample[0] will be passed to the input node as is,
        #if the graph has multiple input tensors,
        #the data part sample[0][i] should be consistent with input_tensors[i] in IR.
        #If the graph has multiple output tensors,
        #the label part sample[1][i] should be consistent with output_tensors[i] in IR.
        sample = [[self.data[idx]], float("-inf")]
        if self.label is not None:
            sample[1] = self.label[idx]
        return sample
```
Dataset plugin的核心是通过`__len__`接口告知OPT对应数据的规模，并通过`__getitem__`接口返回的`sample[0]`作为模型每一次forward运算的输入（用户模型要求的输入数据规范仅用户完全知晓），`sample[1]`作为groundtruth标签信息会由OPT**透传**给相应的Metric plugin（用户模型对应的输出和标签含义信息仅用户完全知晓）。 需要特别说明的是：
- 注册时传入的第一个参数指明了plugin的类别是`PluginType.Dataset`。第二个参数作为版本号，即当有多个同名同类plugin时，会实际调用版本号更大的。
- 实例化Dataset plugin时，会传入两个参数（在cfg文件中用相关字段指定其取值）：`data_file`和`label_file`。这两个参数既可以是实际存放data或label的文件路径，也可以是间接存放一系列其它文件信息的纯文本文件的路径（具体的解析过程全由编写者定制）。
- 所有的数据预处理操作，既可以提前做好后存于硬盘（推荐采用此种方式），实际运行时仅作反序列化读取以加快forward运行，也可以是在`__getitem__`函数内部进行（如存储的数据按NHWC格式，但模型要求NCHW格式，则读取数据后执行相应的permute操作后再填入返回；又如存储的标签索引按从0开始，但模型要求按从1开始，则读取后做相应的offset操作后再填入返回；再如存储的数据未做某种归一化，但模型要求归一化后的数据输入，则读取后做相应的归一化操作后再填入返回）。
- 当模型有多个输入或输出时，`__getitem__`返回的`sample`需与`float IR`中定义的输入输出顺序一致，即`sample[0]`指定的data list需与IR中`input_tensors`顺序一致，`sample[1]`指定的label list需与IR中`output_tensors`顺序一致（若调用的Metric plugin对label数据另有要求，则按该Metric plugin的要求）。

##### Metric plugin编写
Metric plugin需要继承自`OptBaseMetric`类，用@register_plugin(PluginType.Metric, *version*)注册（*version*表示版本号，当有同类同名plugin时会优先调用高版本号的），并实现`__init__, __call__, reset, compute, report`接口。每个接口的含义和编写方式将结合以下代码示例加以说明：
```python
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
import torch

@register_plugin(PluginType.Metric, '1.0')
class TopKMetric(OptBaseMetric):
    #you can pass any string parameters from cfg file, and parse it to what you really want
    #e.g. you can set 'metric = TopKMetric,TopKMetric(5),TopKMetric(10)' in cfg file to enable
    #calculate top1, top5 and top10 accuracy together
    def __init__(self, K='1'):
        self.correct = 0
        self.total = 0
        self.K = int(K)
    #will be called after every batch iteration, the pred is model's output_tensors (the same order in IR),
    #the target is the sample[1] generated by dataset plugin,
    #during quantize_forward the pred will be dequantized before calling metric
    def __call__(self, pred, target):
        _, pt = torch.topk(pred[0].reshape([pred[0].shape[0], -1]), self.K, dim=-1)    #NHWC
        for i in range(target.numel()):
            if target[i] in pt[i]:
                self.correct += 1
        self.total += target.numel()
    #will be called before every epoch iteration to reset the initial state
    def reset(self):
        self.correct = 0
        self.total = 0
    #will be called after every epoch iteration to get the final metric score
    def compute(self):
        try:
            acc = float(self.correct) / float(self.total)
            return acc
        except ZeroDivisionError:
            OPT_ERROR('zeroDivisionError: Topk acc total label = 0')
            return float("-inf")
    #will be called when outputing a string format metric report
    def report(self):
        return "top-%d accuracy is %f" % (self.K, self.compute())

```

需要特别说明的是：
- Metric plugin支持从cfg文件传递构造参数，但仅限于字符串类型，编写plugin时需自行进行字符串类型参数到目标类型的转换。
- 传递给metric plugin的模型输出结果顺序与IR中的output_tensors顺序一致，同时传递的target即为Dataset plugin中给出的sample[1]，pred（quant_forward时会提前自动做反量化）和target之间的对应关系以及计算指标的逻辑完全由用户把控。

#### 代码风格

OPT使用`autopep8`来检查代码风格。`AIPUBuilder/Optimizer/scripts`下有使能自动检查机制的安装脚本，请确保已经安装了`autopep8`并且在当前开发环境可以正常调用。

#### 本地测试

在提交代码前，强烈建议进行一定的本地测试。最直接有效的测试用例可以在[Zhouyi Model Zoo](https://github.com/Arm-China/Model_zoo)中采样并修改复用。如果是修改现有功能或者添加了新功能，记得在对应测试用例的配置文件中使能相应功能。如果是修改或添加算子，记得在所采样或构造的测试用例中包含相应的算子。
