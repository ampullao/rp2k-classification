# 目录

# Wide Residual Networks描述

## 概述

Wide Residual Networks（WRN）是2016年由Sergey Zagoruyko等人提出，从增加网络宽度角度改善ResNet。ResNet通过增加深度提高了精确度，但随着网络的加深，精度的提升也逐渐减小，网络的训练速度也变得非常慢。所以Sergey Zagoruyko等人提出了WRN，并通过实验证明了增加网络的宽度能够提高网络的性能，在相同数量的参数下WRN的训练速度比ResNet快。

## 论文

1. [论文](https://arxiv.org/pdf/1605.07146.pdf):Sergey Zagoruyko, Nikos Komodakis."Wide Residual Networks"

## 模型架构

WRN的总体网络架构如下:[链接](https://arxiv.org/pdf/1605.07146.pdf)

## 数据集

使用的数据集：[RP2K](https://www.pinlandata.com/rp2k_dataset/)

- 数据集大小：共2388个类，大小不一的彩色图像
    - 训练集：344,854张图像
    - 测试集：39,457张图像
- 数据格式：JEPG, PNG
    - 注：数据在dataset.py中处理。
- 下载数据集。目录结构如下：
```text
└─all
   ├─train # 训练数据集
   └─test  # 评估数据集
```

## 环境要求

- 硬件(Ascend)
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

> - <font size=2>训练时，DATASET_PATH={RP2K路径}/all/train;</font>
> - <font size=2>评估和推理时，DATASET_PATH={RP2K路径}/all/test;</font>

- Ascend处理器环境运行

```text
# 单机训练
用法：bash run_standalone_train.sh [DATASET_PATH]  [CONFIG_PATH] [RESUME_CKPT]（可选）

# 运行评估示例
用法：bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]

```

## 脚本说明

### 脚本和样例代码

```text
.
└──WRN
  ├── README.md
  ├── config                              # 参数配置
    └── wrn50-2_config.yaml
  ├── scripts
    ├── run_eval.sh                        # 启动Ascend评估
    └── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
  ├── src
    ├── dataset.py                         # 数据预处理
    └── wideresnet.py                      # WRN50-2网络
    ├── model_utils
       └── config.py                       # 参数配置
  ├── requirements.txt                     # 第三方依赖
  ├── eval.py                              # 评估网络
  └── train.py                             # 训练网络
```

### 脚本参数

```text
# Path for local
data_path: "../data"               # 数据集目录
ckpt_path: "../checkpoint_path/"   # 模型权重目录
checkpoint_name: ""                # 模型权重文件名
summary_base_dir: "../summary_dir" # MindSpore Insight的Summary日志文件

# Training options
batch_size: 128                    # 输入张量的批次大小
lr: 0.1                            # 学习率
momentum: 0.9                      # 动量
epoch_size: 20                     # 此值仅适用于训练
```

## 训练过程

> 提供训练信息，区别于quick start，此部分需要提供除用法外的日志等详细信息

### 训练

> 提供训练脚本的使用方法

例如：在昇腾上使用分布式训练运行下面的命令

```shell
bash run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

> 提供训练过程日志

```log
# grep "loss is " train.log
epoch:1 step:390, loss is 1.4842823
epcoh:2 step:390, loss is 1.0897788
```

> 提供训练结果日志
例如：训练checkpoint将被保存在`XXXX/ckpt_0`中，你可以从如下的log文件中获取结果

```log
epoch: 11 step: 7393 ,rpn_loss: 0.02003, rcnn_loss: 0.52051, rpn_cls_loss: 0.01761, rpn_reg_loss: 0.00241, rcnn_cls_loss: 0.16028, rcnn_reg_loss: 0.08411, rcnn_mask_loss: 0.27588, total_loss: 0.54054
epoch: 12 step: 7393 ,rpn_loss: 0.00547, rcnn_loss: 0.39258, rpn_cls_loss: 0.00285, rpn_reg_loss: 0.00262, rcnn_cls_loss: 0.08002, rcnn_reg_loss: 0.04990, rcnn_mask_loss: 0.26245, total_loss: 0.39804
```

### 迁移训练（可选）

> 提供如何根据预训练模型进行迁移训练的指南

### 分布式训练

> 同上

## 评估

### 评估过程

> 提供eval脚本用法

### 评估结果

> 提供推理结果

例如：上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```log
accuracy:{'acc':0.934}
```

## 导出

### 导出过程

> 提供export脚本用法

### 导出结果

> 提供export结果日志

## 推理

### 推理过程

> 提供推理脚本

```bash
bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
```

### 推理结果

> 提供推理结果

## 性能

### 训练性能

提供您训练性能的详细描述，例如finishing loss, throughput, checkpoint size等

你可以参考如下模板

| Parameters                 | Ascend 910                                                   | GPU |
| -------------------------- | ------------------------------------------------------------ | ----------------------------------------------|
| Model Version              | ResNet18                                                     |  ResNet18                                     |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |  PCIE V100-32G                                |
| uploaded Date              | 02/25/2021 (month/day/year)                                  | 07/23/2021 (month/day/year)                   |
| MindSpore Version          | 1.1.1                                                        | 1.3.0                                         |
| Dataset                    | CIFAR-10                                                     | CIFAR-10                                      |
| Training Parameters        | epoch=90, steps per epoch=195, batch_size = 32               | epoch=90, steps per epoch=195, batch_size = 32|
| Optimizer                  | Momentum                                                     | Momentum                                      |
| Loss Function              | Softmax Cross Entropy                                        | Softmax Cross Entropy                         |
| outputs                    | probability                                                  | probability                                   |
| Loss                       | 0.0002519517                                                 |  0.0015517382                                 |
| Speed                      | 13 ms/step（8pcs）                                           | 29 ms/step（8pcs）                            |
| Total time                 | 4 mins                                                       | 11 minds                                      |
| Parameters (M)             | 11.2                                                         | 11.2                                          |
| Checkpoint for Fine tuning | 86M (.ckpt file)                                             | 85.4 (.ckpt file)                             |
| Scripts                    | [link](https://gitee.com/mindspore/models/tree/master/official/cv/)                       |

### 推理性能

> 提供推理性能的详细描述，包括耗时，精度等

你可以参照如下模板

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNet18                    |
| Resource            | Ascend 910; OS Euler2.8     |
| Uploaded Date       | 02/25/2021 (month/day/year) |
| MindSpore Version   | 1.1.1                       |
| Dataset             | CIFAR-10                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 94.02%                      |
| Model for inference | 43M (.air file)             |

## 随机情况说明

> 说明该项目有可能出现的随机事件

## 参考模板

此部分不需要出现在你的README中
[resnet_readme](https://gitee.com/mindspore/models/blob/master/official/cv/ResNet/README_CN.md)

## 贡献指南

如果你想参与贡献昇思的工作当中，请阅读[昇思贡献指南](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING_CN.md)和[how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

### 贡献者

此部分根据自己的情况进行更改，填写自己的院校和邮箱

* [c34](https://gitee.com/c_34) (Huawei)

## ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。
