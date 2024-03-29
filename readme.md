[TOC]

## Compass Optimizer
Compass Optimizer (**OPT** for short), is part of the Zhouyi Compass Neural Network Compiler (python package name is AIPUBuilder). The OPT is designed for converting the float Intermediate Representation (IR) generated by the [Compass Unified Parser](https://github.com/Arm-China/Compass_Unified_Parser) to an optimized quantized or mixed IR (through techniques like quantization and graph optimization) which is suited for Zhouyi NPU hardware platforms. You can find more about Compass IR and Compass SDK [here](https://aijishu.com/a/1060000000215443).

In addition to this readme file, you can find a more detailed tutorial document (`tutorial.pdf`).

### Main Features
- Multiple quantization methods: per-tensor quantization, per-channel quantization, asymmetric quantization, and symmetric quantization.
- Mixed precision quantization: multiple different quantization bitwiths on different layers, quantized layers mixed with original float point layers, automatically searching quantization bits configuration.
- Support for setting per-layer quantization parameters through JSON configuration file.
- Multiple common calibration strategies:

  - Averaging: Computes an weighted average of the statistic values of different calibration batches.

  - Mean +/- N*Std: Takes N standard deviations for the tensor's mean, where N is configurable.

  - ACIQ: Refer to [Post-training 4-bit quantization of convolution networks for rapid-deployment](https://arxiv.org/abs/1810.05723).

  - KLD: Refer to [8-bit inference with TensorRT](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf).

  - EasyQuant: Refer to [EasyQuant: Post-training Quantization via Scale Optimization.](https://arxiv.org/abs/2006.16669)

  - Adaround: Refer to [Up or Down? Adaptive Rounding for Post-Training Quantization.](https://arxiv.org/abs/2004.10568)

  - Adaquant: Refer to [Improving Post Training Neural Quantization: Layer-wise Calibration and Integer Programming](https://arxiv.org/abs/2006.10518)

- Compatible with all Zhouyi NPU series processors, specially optimizing the quantization schemes and the whole computation graph.


### Quick Start

#### Installation instructions
You can build AIPUBuilder by yourself with [Compass_Integration](https://github.com/Arm-China/Compass_Integration), and find more documents in [MiniPkg](https://aijishu.com/a/1060000000215443). After installing the AIPUBuilder, the OPT is available.

The OPT can also run independently. You can run `AIPUBuilder/Optimizer/tools/optimizer_main.py` with the following requirements.
#### Requirements
* Python3 >= 3.8.5
* NumPy >= 1.22.3
* NetworkX >= 2.8
* torch >= 1.11.1
* torchvision >= 0.12.0
#### Running the OPT
The OPT uses a txt config file (.cfg) as the input. You can run it with your config file as follows：
```bash
export PYTHONPATH=./:$PYTHONPATH
python3 ./AIPUBuilder/Optimizer/tools/optimizer_main.py --cfg ./opt.cfg
```
#### Config file format
All options are under the `Common` section:
- `graph` is the path for this model's float IR definition file

- `bin` is the path for this model's float IR weights binary file

- `model_name` is the name of the model

- `dataset` is the name of the dataset plugin for this model's input dataset (you can use `optimizer_main.py --plugin` to check all the accessible plugins)

- `calibration_data` is the path of the dataset used for calibration during quantization

- `calibration_batch_size` is the `batch_size` used for calibration during quantization

- `metric` is the name of metric plugins for computing accuracy metrics for this model (you can use `optimizer_main.py --plugin` to check all the accessible plugins). If omitted, it will not compute accuracy metrics

- `data`: If `metric` is set, then assign the path of the corresponding dataset

- `label`: If `metric` is set, then assign the path of the corresponding labels

- `metric_batch_size`: If `metric` is set, then assign the corresponding `batch_size`

- `quantize_method_for_weight` is the quantization method used for weights, such as:

  - `per_tensor_symmetric_restricted_range`

  - `per_tensor_symmetric_full_range`

  - `per_channel_symmetric_restricted_range`

  - `per_channel_symmetric_full_range`

  It defaults to `per_tensor_symmetric_restricted_range`.

- `quantize_method_for_activation` is the quantization method used for weights, such as:

  - `per_tensor_symmetric_restricted_range`

  - `per_tensor_symmetric_full_range`

  - `per_tensor_asymmetric`

  It defaults to `per_tensor_symmetric_full_range`.

- `weight_bits` are the bits used for quantizing weight tensors. This parameter defaults to 8

- `bias_bits` are the bits used for quantizing bias tensors. This parameter defaults to 32

- `activation_bits` are the bits used for quantizing activation tensors. This parameter defaults to 8

- `lut_items_in_bits` are maximal LUT items (in bits, as only support LUT with 2**N items) amount when representing nonlinear functions in quantization. This parameter defaults to 8. It is suggested to set to 10+ when quantizing activations to 16bit

- `output_dir` is the output directory path

You can use `optimizer_main.py --field` to check more about all configurable fields.

#### Running Tests

##### Test Models

You can find an example case in `AIPUBuilder/Optimizer/test/model_test/squeezenet` (just execute `sh ./run.sh`), and you can find more model examples in [Zhouyi Model Zoo](https://github.com/Arm-China/Model_zoo).

##### Testing Operators

A single operator's test case is treated as a special model test case. You can also find an example case in `AIPUBuilder/Optimizer/test/op_test` (just execute `sh ./run.sh`).

## Process Flow and Design Philosophy

The following is the process flow of the OPT:

![opt flow](./images/opt_flow.svg)

1. Read the IR generated by [Parser](https://github.com/Arm-China/Compass_Unified_Parser), and construct an internal graph representation `g`.
2. Apply a forward on `g` with all zeros as inputs to check validity.
3. Perform pre-quantization graph optimization on `g`.
4. Apply forwards with the given calibration dataset to collect statistic information on tensors in `g`.
5. Perform quantization aware graph optimization on `g.`
6. Apply quantization with given configurations and generate new graph `qg`.
7. Perform post-quantization graph optimization on `qg.`
8. Apply a forward on `qg` with all zeros as inputs to check validity.
9. Output serialized IR of `qg.`
10. Optionally dump internal tensors and compute metrics.

The OPT takes a mechanism that separates operators' implementation from the scheduling framework. Each operator's computation and quantization procedure, the model's input data feeding (dataset parsing) and output data processing (metric), are all integrated as plugins and scheduled by the `OptMaster` class. So it is convenient for third-party developers to porting their own particular models.

### Development Guide

#### Core Data Structures
The following are some core data structures of the OPT:
![opt uml](./images/opt_uml.svg)

- `Dtype` defines basic data types that may occur in the Compass IR
- `PyTensor` is the tensor wrapper class in OPT. It is actually stored and calculated through `torch.Tensor`
- `PyNode` represents the layer concept in NN models. Connections between layers are represented as shared tensors (the `PyTensor` instances stored in each layer's `inputs` and `outputs`)
- `PyGraph` represents the model's computation graph. Its network structure is maintained internally through a `networkx.DiGraph` instance. `QuantizeGraph` is inherited from `PyGraph`, and held by `OptMaster`
- `OptMaster` controls the whole process flow, instances model's corresponding dataset plugin, and metric plugins according to the configuration file

#### Plugins Development
The most commonly used development paradigm is to add more plugins of Operator, Dataset, or Metric to support special models, so this section will focus on plugin development.
##### Naming
The following prefix names are suggested to use:
- aipubt_opt_op_ for the optimizer operator plugin.
- aipubt_opt_dataset_ for the dataset plugin of the optimizer.
- aipubt_opt_metric_ for the metric plugin of the optimizer.
##### Search Paths
Plugin files will be searched in the following locations:
1. Paths defined in the environment variable `AIPUPLUGIN_PATH`, for example：
    `export AIPUPLUGIN_PATH=/home/user/aipubuilder_plugins/:$AIPUPLUGIN_PATH`
2. The current working directory, which is `./plugin/`.
##### Operator Plugin

The operator plugin needs to implement two methods — forward and quantize：

- Use `op_register(OpType, version)` to register a forward function `forward_function_name(self, *args)`.
- Use `quant_register(OpType, version)` to register a quantize function `quantize_function_name(self, *args)`.

Where `OpType` is the data structure that holds the type of operators. If you want to replace the existing operator's implementation, just set the corresponding operator type OpType.*layer_type_name*, and set the version greater than 1.0 (the existing inner operator's version is 1.0) when calling the register functions. If you want to implement a new operator type, you need to globally call the `register_optype('new_layer_type_name')` function to register the new operator type to `OpType`, and then use the OpType.*new_layer_type_name*. When there exist more than one implementations of the same operator type, the one with the highest version number will be called. Because the forward function and the quantize function are registered with different APIs, so if you want to completely replace an existing operator, you need to offer both higher version (forward and quantize) functions. `self` is pointed to a `PyNode` instance (represents a layer in IR). Its usage is introduced through the following demo:

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

In addition, the OPT will execute a forward pass after parsing float IR to ensure that each operator's forward function always has been called at least once before its quantize function is called (conversely is not guaranteed, which means that after each operator's quantize function has been called, its forward function may not be called subsequently). Therefore, the variables in placeholder or attrs set in the forward function is ensured to be visible in the quantize function. You can find more meaningful and specific demos under `AIPUBuilder/Optimizer/ops`.

##### Dataset Plugin
Dataset plugin inherits from the `torch.utils.data.Dataset` class. It should be implemented with three methods `__init__`, `__len__` and `__getitem__`. The following is a simple example of a `NumpyDataset`:
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
Dataset plugin uses the `__len__` API to tell the amount of corresponding dataset, and uses `__getitem__` to return a specific sample. `sample[0]` stores the input data that forwards computation needs (you must know the data format that your model needs). `sample[1]` stores the groundtruth labels which will be passed to the corresponding metric plugin (you must know the correct relationship between your model's output tensors and corresponding labels). Additionally:
- The first parameter sent to the register function identifies the plugin type (`PluginType.Dataset`). The second parameter is the version number (When there exist more than one implementations, the one with the highest version number will be called).
- When a dataset plugin is instanced, two parameters will be passed to it：`data_file` and `label_file`. The two parameters in the cfg file may be specific paths of the data file and label file, also they can be files that record the real paths of specific data or label files. The parsing procedure is totally controlled by the developer.
- All the preprocesses can be performed offline (which is recommended), and also can be executed in the `__getitem__` function (for example, normalization steps, layout converting, and so on).
- If the graph has multiple input tensors, the data part `sample[0][i]` should be consistent with `input_tensors[i]` in IR. If the graph has multiple output tensors, the label part `sample[1][j]` should be consistent with `output_tensors[j]` in IR.

##### Metric Plugin
Metric plugin inherits from the `OptBaseMetric` class. You need to use @register_plugin(PluginType.Metric, *version*) to register (when there exist more than one implementations, the one with the highest version number will be called), and implement these interfaces: `__init__`, `__call__`, `reset`, `compute`, and `report`. The following is a simple example of a `TopKMetric`:
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

**Note**:

- A metric plugin can get initial parameters from the cfg file, but it only supports string type, so you need to parse these parameters from the string in the `__init__` function.
- If the model has multiple outputs, the order of `pred` outputs is aligned with the order of ‘output_tensors’ in the header of IR.
- The `pred` and `target` passed by the OPT will be automatically dequantized in advance when running a quantized inference forward.

#### Code Style

The OPT uses `autopep8` for code format checking. `AIPUBuilder/Optimizer/scripts` offers related git hooks. Ensure that you have installed `autopep8` and it is available in your environment.

#### Local Test

Before pushing your codes, it is strongly recommended running enough local test cases to verify them. The most effective way is to sample cases from [Zhouyi Model Zoo](https://github.com/Arm-China/Model_zoo) and have them all passed. If you modified or added some features, remember to trigger them in the configuration file. If you added or modified the implementations of some operators, remember to ensure that models that hold these operators were included in your test cases.

