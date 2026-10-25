# MTA-Net: Grain-Size Composition Prediction from Sedimentary Simulation Experiment Images

[English](#about-mta-net) | [中文](#中文说明)

## About MTA-Net

MTA-Net predicts grain-size composition from images of sedimentary simulation experiments. It combines a ResNet34 backbone with horizontally enhanced multiscale local binary patterns (HMS-LBP), texture-guided feature modulation and the convolutional block attention module (CBAM).

For each image, the model estimates a five-component vector of grain-size volume fractions:

| Component | Grain-size interval |
| --- | --- |
| 1 | <100 μm |
| 2 | 100–250 μm |
| 3 | 250–500 μm |
| 4 | 500–1000 μm |
| 5 | 1000–2000 μm |

The predicted fractions are non-negative and sum to one. The task is composition regression; the output components are proportions rather than mutually exclusive class labels.

This repository provides the model implementation, HMS-LBP and CBAM modules, training and testing scripts, and evaluation metric calculations associated with the MTA-Net study.

## Repository Structure

```text
MTA-Net/
├── Parameters.py
├── Prepare_Data.py
├── train_MTA_Net.py
├── test_MTA_Net.py
├── requirements.txt
├── Datasets/
│   ├── __init__.py
│   └── fixed_split_seed21.json
├── Utils/
│   ├── __init__.py
│   ├── MTA_Net.py
│   ├── HMS_LBP.py
│   ├── Texture_Injection.py
│   ├── CBAM.py
│   ├── Network_functions.py
│   ├── Save_Results.py
│   └── Baseline_models.py
├── Experiments/
│   └── ...
└── samples/
    ├── images/
    │   ├── wh1-001.tif
    │   ├── ...
    │   └── wh1-202.tif
    ├── labels.csv
    ├── sample_manifest.csv
    └── README.md
```

| File or directory | Purpose |
| --- | --- |
| `Parameters.py` | Model settings, training parameters and default paths |
| `Prepare_Data.py` | Data preparation and data-loader construction |
| `Utils/MTA_Net.py` | MTA-Net architecture |
| `Utils/HMS_LBP.py` | HMS-LBP feature extraction and padding-mask handling |
| `Utils/Texture_Injection.py` | Texture projection and feature modulation |
| `Utils/CBAM.py` | Channel and spatial attention |
| `Utils/Network_functions.py` | Model initialization, loss, training, evaluation and metrics |
| `Utils/Save_Results.py` | Checkpoint, training-history, prediction and metric saving |
| `train_MTA_Net.py` | Single-seed model training |
| `test_MTA_Net.py` | Evaluation of a specified checkpoint on a labelled test set |
| `Datasets/fixed_split_seed21.json` | Fixed training, validation and test sample IDs |
| `Experiments/` | Additional experiment configurations and components |
| `samples/` | Image data, measured labels and sample information |

## Installation

Use Python 3.11. The main dependencies are PyTorch, torchvision, NumPy, pandas, Pillow and openpyxl.

Install a PyTorch/torchvision build appropriate for your CPU or CUDA environment, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Additional dependencies for the experiment components are listed in `Experiments/requirements-experiments.txt`.

The ResNet34 backbone uses ImageNet initialization during training. On first use, torchvision downloads the corresponding weights unless they are already cached. These ImageNet weights are distinct from a trained MTA-Net task checkpoint.

## Dataset and Labels

The study dataset consists of 202 sediment images paired with measured grain-size compositions. Organize the images and labels under `samples/` as shown above.

The label file uses the following columns:

```text
Sample_ID,lt100,100_250,250_500,500_1000,1000_2000
```

The five numerical columns store measured volume percentages. Model targets use the corresponding fractions normalized to sum to one. Match each image to its label using the sample identifier.

The main fixed split contains 145 training samples, 26 validation samples and 31 test samples. Its membership is specified in `Datasets/fixed_split_seed21.json`.

Image–label correspondence and original subset information are recorded in `samples/sample_manifest.csv`. See `samples/README.md` for dataset-specific information.

## Train MTA-Net

Run the following command from the repository root:

```bash
python train_MTA_Net.py --image-dir samples/images --labels samples/labels.csv --split Datasets/fixed_split_seed21.json --output results/seed21 --seed 21
```

Training settings are defined in `Parameters.py`. The defaults include a batch size of 8, 50 epochs and the AdamW optimizer with a learning rate of 0.0005 and weight decay of 0.0001.

Each invocation trains one model seed. Repeated runs on the same fixed split use different `--seed` values; changing the model seed does not require regenerating the split.

To inspect the available command-line options:

```bash
python train_MTA_Net.py --help
```

## Test MTA-Net

Evaluate a checkpoint produced by training:

```bash
python test_MTA_Net.py --checkpoint results/seed21/best_checkpoint.pt --image-dir samples/images --labels samples/labels.csv --split Datasets/fixed_split_seed21.json --output results/test_seed21
```

Replace the checkpoint path with the actual path produced by your training run. Testing requires the measured labels to calculate evaluation metrics.

Training and evaluation outputs include checkpoints, training records, sample-level predictions and evaluation metrics. Output handling is implemented in `Utils/Save_Results.py`.

## Evaluation Metrics

The metric calculations are implemented in `distribution_metrics` in `Utils/Network_functions.py`. All metrics use volume fractions on a 0–1 scale.

For N samples and five grain-size intervals, with measured fraction y and predicted fraction p:

```text
χ² distance = mean_i [0.5 × sum_j ((p_ij − y_ij)² / (p_ij + y_ij + ε))]

MAE = mean_j [mean_i (|p_ij − y_ij|)]

RMSE_j = sqrt(mean_i ((p_ij − y_ij)²))

NRMSE_j (%) = RMSE_j / (max_i(y_ij) − min_i(y_ij) + ε) × 100

Overall NRMSE (%) = mean_j (NRMSE_j)
```

Here, ε = 1e-8. Each normalization range is calculated from the measured values in the evaluation set. Overall NRMSE is the unweighted mean across the five intervals.

For repeated runs, compute the overall metrics separately for each model seed, then report their mean and sample standard deviation. The aggregation utilities are provided in `Experiments/statistics.py`.

## Additional Experiments

The `Experiments/` directory contains configurations and components for model comparisons, ablation studies, LBP encoding variants, neighbourhood-radius settings, feature-injection stages and profile-segment-grouped validation.

Refer to [Experiments/README.md](Experiments/README.md) for the available components, settings and experiment-specific instructions.

---

# 中文说明

## 关于MTA-Net

MTA-Net用于根据沉积物理模拟实验图像预测粒度组成。模型将ResNet34主干、水平增强多尺度局部二值模式（HMS-LBP）、纹理引导的特征调制及卷积块注意力模块（CBAM）相结合。

每张图像对应一个五维粒度体积分数组成向量，依次表示：<100 μm、100–250 μm、250–500 μm、500–1000 μm和1000–2000 μm。预测比例非负且总和为1。该任务属于组成回归，五个输出表示各粒级的比例，而非互斥的分类标签。

本仓库提供MTA-Net研究对应的模型实现、HMS-LBP与CBAM模块、训练与测试脚本，以及评价指标计算代码。

## 文件功能

上方目录结构展示了项目的主要文件。各文件和目录的作用如下：

| 文件或目录 | 功能 |
| --- | --- |
| `Parameters.py` | 模型设置、训练参数和默认路径 |
| `Prepare_Data.py` | 数据准备与数据加载器构建 |
| `Utils/MTA_Net.py` | MTA-Net模型架构 |
| `Utils/HMS_LBP.py` | HMS-LBP特征提取及填充区域掩膜处理 |
| `Utils/Texture_Injection.py` | 纹理特征投影与特征调制 |
| `Utils/CBAM.py` | 通道注意力和空间注意力 |
| `Utils/Network_functions.py` | 模型初始化、损失、训练、评价及指标计算 |
| `Utils/Save_Results.py` | 保存权重、训练记录、预测结果及指标 |
| `train_MTA_Net.py` | 单个模型种子的训练入口 |
| `test_MTA_Net.py` | 使用指定权重评价带标签的测试集 |
| `Datasets/fixed_split_seed21.json` | 固定训练集、验证集和测试集的样本ID |
| `Experiments/` | 其他实验的配置与组件 |
| `samples/` | 图像、粒度实测标签及样本信息 |

## 环境安装

使用Python 3.11。主要依赖包括PyTorch、torchvision、NumPy、pandas、Pillow和openpyxl。

先根据计算机的CPU或CUDA环境安装匹配的PyTorch和torchvision，再在项目根目录运行：

```bash
python -m pip install -r requirements.txt
```

其他实验组件的依赖列于 `Experiments/requirements-experiments.txt`。

训练时，ResNet34主干使用ImageNet预训练初始化。若本地尚未缓存相应权重，torchvision会在首次使用时下载。这些ImageNet预训练权重与训练得到的MTA-Net任务权重不同。

## 数据与标签

研究数据集由202张沉积物图像及其对应的粒度实测组成构成。按照上方目录结构，将图像和标签放在 `samples/` 下。

图像文件位于 `samples/images/`，命名为 `wh1-001.tif` 至 `wh1-202.tif`。标签文件 `samples/labels.csv` 的列名为：

```text
Sample_ID,lt100,100_250,250_500,500_1000,1000_2000
```

五个数值列保存实测体积百分数，模型标签使用归一化后总和为1的组成比例。通过样本ID匹配图像和标签。

主实验采用固定划分：训练集145个、验证集26个、测试集31个样本，具体归属见 `Datasets/fixed_split_seed21.json`。

图像与标签对应关系及原始子集信息记录于 `samples/sample_manifest.csv`。数据集详细说明见 `samples/README.md`。

## 训练MTA-Net

在项目根目录运行：

```bash
python train_MTA_Net.py --image-dir samples/images --labels samples/labels.csv --split Datasets/fixed_split_seed21.json --output results/seed21 --seed 21
```

训练设置见 `Parameters.py`。默认batch size为8，训练50轮，采用AdamW优化器，学习率为0.0005，权重衰减为0.0001。

每次运行训练一个模型种子。在同一固定划分上进行重复实验时，修改 `--seed` 即可，无需重新生成数据划分。

查看命令行参数：

```bash
python train_MTA_Net.py --help
```

## 测试MTA-Net

使用训练得到的权重进行评价：

```bash
python test_MTA_Net.py --checkpoint results/seed21/best_checkpoint.pt --image-dir samples/images --labels samples/labels.csv --split Datasets/fixed_split_seed21.json --output results/test_seed21
```

将权重路径替换为实际训练输出的路径。测试时需要实测标签，以计算评价指标。

训练和评价输出包括权重、训练记录、逐样本预测及评价指标，保存功能位于 `Utils/Save_Results.py`。

## 评价指标

三项评价指标为χ²距离、MAE和NRMSE，均使用0–1范围的体积分数组成。计算函数为 `Utils/Network_functions.py` 中的 `distribution_metrics`，公式见上方英文部分。

χ²距离先对各样本的五个粒级求和，再对样本取平均；MAE对样本和五个粒级的绝对误差取平均。NRMSE先将各粒级RMSE除以当前评价集该粒级实测值的极差并换算为百分数，再对五个粒级等权平均。数值稳定项ε为1e-8。

重复实验应先分别计算每个模型种子的整体指标，再汇总均值和样本标准差。汇总工具见 `Experiments/statistics.py`。

## 其他实验

`Experiments/` 提供模型对比、消融、LBP编码、邻域半径、特征注入阶段及剖面区段分组验证相关的配置与组件。

具体组件、设置及实验说明见 [Experiments/README.md](Experiments/README.md)。

