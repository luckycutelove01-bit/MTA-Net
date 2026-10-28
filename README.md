# MTA-Net: Grain-Size Composition Prediction from Sedimentary Simulation Experiment Images

[English](#overview) | [中文](#中文说明)

## Overview

MTA-Net estimates five grain-size volume fractions from sedimentary simulation experiment images. This repository provides the MTA-Net architecture, HMS-LBP descriptor, texture modulation, CBAM, training, checkpoint evaluation and distribution metrics for the study **“MTA-Net for Predicting Grain-Size Composition Distributions from Sedimentary Simulation Experiment Images by Integrating Multiscale Texture Features.”**

The dataset is included: **202 RGB TIFF images and 202 measured label vectors**, matched by `Sample_ID`.

| Output | Grain-size interval |
|---|---|
| 1 | <100 μm |
| 2 | 100–250 μm |
| 3 | 250–500 μm |
| 4 | 500–1000 μm |
| 5 | 1000–2000 μm |

The model returns five logits; the loss and evaluation functions apply softmax to obtain non-negative fractions summing to one. This is composition regression.

## File Guide

| Path | Function |
|---|---|
| `Parameters.py` | Architecture, optimization and default dataset settings |
| `Prepare_Data.py` | Validate fixed splits, sample batches and construct loaders |
| `Datasets/read_labels.py` | Read labels, check percentages and IDs, match images |
| `Datasets/data_loader.py` | Prepare RGB, masked HMS-LBP and normalized target tensors |
| `Datasets/fixed_split_seed21.json` | Main split: 145 training / 26 validation / 31 test |
| `Utils/MTA_Net.py` | ResNet34-based MTA-Net |
| `Utils/HMS_LBP.py` | Horizontal neighbours, uniform-u2 lookup, masks and histograms |
| `Utils/Texture_Injection.py` | Texture projection and residual channel modulation |
| `Utils/CBAM.py` | Channel attention followed by spatial attention |
| `Utils/Network_functions.py` | Loss, training, evaluation, checkpoint loading and metrics |
| `Utils/Save_Results.py` | Save checkpoints, histories, predictions and metrics |
| `Utils/Baseline_models.py` | Shallow CNN and torchvision baseline regressors |
| `train_MTA_Net.py` | Train one MTA-Net model seed |
| `test_MTA_Net.py` | Evaluate an MTA-Net checkpoint |
| `verify_dataset.py` | Check dataset IDs, decoding, hashes and split coverage |
| `tests/test_pipeline.py` | Dataset, texture, metrics, backward-pass and checkpoint tests |
| `samples/` | Images, measured labels and [dataset documentation](samples/README.md) |
| `Experiments/` | [Model comparison, ablation, encoding, radius, stage and grouped training/evaluation](Experiments/README.md) |
| `SOURCE_MANIFEST.json` | SHA256 inventory of distributed source files |

## Environment

The manuscript reports training with Python 3.11. Create a dedicated environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Choose matching PyTorch/torchvision CPU or CUDA builds. ResNet34 training uses ImageNet initialization; torchvision downloads these initialization weights on first use unless they are cached. Training produces a task checkpoint for subsequent evaluation.

Additional experiment components use scikit-learn:

```bash
python -m pip install -r Experiments/requirements-experiments.txt
```

## Dataset Verification

Run commands from the repository root:

```bash
python verify_dataset.py
```

This decodes all 202 images, checks unique IDs and SHA256 values against `samples/sample_manifest.csv`, validates all measured label vectors and checks disjoint coverage of both fixed splits.

`labels.csv` columns:

```text
Sample_ID,lt100,100_250,250_500,500_1000,1000_2000
```

IDs include the image extension, for example `wh1-001.tif`. Numeric values are measured **volume percentages**. The loader divides each row by its total to obtain fractions; the CSV values remain unchanged. Small rounding differences around 100% are retained in the published labels.

## Model and Training Settings

| Setting | Value |
|---|---|
| RGB input | Preserve aspect ratio, resize and grey-pad to 224 × 224 |
| RGB normalization | ImageNet mean/std |
| Training augmentation | Brightness, contrast, saturation jitter: 0.1 each |
| HMS input | Original RGB → float luminance → resize and padding |
| HMS radii | {1,2}, descriptors averaged across radii |
| HMS histogram | 59 uniform-u2 bins × 4 × 4 regions = **944 features** |
| Padding mask | Count centres whose neighbours all lie within valid content |
| Texture injection | After Stage2, projection 944 → 32 → 128 |
| Modulation | `F + gamma * (F * modulation)`, learnable gamma initialized at 0.1 |
| CBAM | After Stage4; reduction 16, spatial kernel 7 |
| Prediction head | 512 → 512 → 256 → 5, dropout 0.3/0.2 |
| Optimizer | AdamW, learning rate 0.0005, weight decay 0.0001 |
| Batch size / epochs | 8 / 50 |
| Scheduler | ReduceLROnPlateau on validation loss; factor 0.5, patience 8 |
| Checkpoint | Lowest validation χ², then validation MAE to break ties |
| Loss | CDF/L1/cosine weights 0.6/0.3/0.1; entropy coefficient 0.6 |
| Model seeds | 21,42,63,84,105 |

The plain ResNet34 baseline and every ResNet34 ablation use the same MTA-Net prediction head, including BatchNorm, ReLU and dropout (0.3/0.2). Comparison, ablation and profile validation share this ResNet34 baseline.

Colour jitter affects the RGB branch only. Validation/test preprocessing is deterministic. No geometric augmentation is applied. A singleton training tail is rebalanced into non-singleton batches for BatchNorm, retaining every training sample; shuffling changes with the epoch.

## Train MTA-Net

```bash
python train_MTA_Net.py --seed 21 --output results/seed21
```

Equivalent command with explicit paths:

```bash
python train_MTA_Net.py --image-dir samples/images --labels samples/labels.csv --split Datasets/fixed_split_seed21.json --seed 21 --output results/seed21
```

Use `--device cpu` or `--device cuda`; the default is automatic. `--epochs` and `--batch-size` are also available. Remaining defaults are in `Parameters.py`.

Five model seeds, PowerShell:

```powershell
foreach ($modelSeed in 21,42,63,84,105) {
    python train_MTA_Net.py --seed $modelSeed --output "results/seed$modelSeed"
    if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $modelSeed" }
}
```

Bash:

```bash
for model_seed in 21 42 63 84 105; do
    python train_MTA_Net.py --seed "$model_seed" --output "results/seed$model_seed" || exit 1
done
```

The fixed dataset membership stays unchanged across model seeds. Use empty or new output directories.

## Evaluate a Checkpoint

```bash
python test_MTA_Net.py --checkpoint results/seed21/best_checkpoint.pt --output results/test_seed21
```

Evaluation checks test membership/order against the checkpoint. Measured labels are used for metrics. A checkpoint stores the model state, method settings, run parameters, selected epoch, validation metrics and split IDs.

| Output | Contents |
|---|---|
| `best_checkpoint.pt` | Selected model checkpoint, produced by training |
| `training_history.csv` | Epoch losses, validation metrics and learning rate |
| `run_parameters.json` | Run configuration |
| `method_settings.json` | Architecture/preprocessing constants |
| `split_ids.json` | Dataset memberships |
| `test_predictions.csv` | IDs, five true fractions, five predicted fractions |
| `test_metrics.json` | Test loss and overall/per-interval metrics |
| `per_interval_metrics.csv` | Per-interval χ² contributions, MAE and NRMSE |
| `best_epoch.json` | Selected epoch |

The standalone test command additionally writes `evaluation_source.json`.

## Evaluation Metrics

`distribution_metrics` in `Utils/Network_functions.py` takes measured fractions `y` and predicted fractions `p`, both shaped `[N,5]`:

```text
χ² = mean_i [0.5 * sum_j ((p_ij-y_ij)^2 / (p_ij+y_ij+epsilon))]
MAE = mean_j [mean_i abs(p_ij-y_ij)]
RMSE_j = sqrt(mean_i ((p_ij-y_ij)^2))
NRMSE_j (%) = 100 * RMSE_j / (max_i y_ij - min_i y_ij + epsilon)
Overall NRMSE (%) = mean_j NRMSE_j
epsilon = 1e-8
```

NRMSE uses the measured range of each interval in the evaluated subset and averages the five interval values equally. Per-interval χ² entries contribute to the same overall χ² definition. Aggregate each run's overall metrics using the arithmetic mean and **sample standard deviation (`ddof=1`)**. `Experiments/statistics.py` validates all five seed IDs before aggregation.

## Profile-Segment Split and Other Components

`Experiments/profile_split_C.json` specifies 143/20/39 samples. To run the main MTA-Net pipeline with these memberships:

```bash
python train_MTA_Net.py --split Experiments/profile_split_C.json --seed 21 --output results/profile_seed21
python test_MTA_Net.py --checkpoint results/profile_seed21/best_checkpoint.pt --split Experiments/profile_split_C.json --output results/profile_test_seed21
```

[Experiments/README.md](Experiments/README.md) documents complete training, checkpoint evaluation and five-seed summary commands for comparisons, ablations, LBP encodings, radii, insertion stages and profile validation. Neural configurations share the root training/loss/metric functions; forests are fitted on training IDs only. The root training/testing scripts instantiate the default `Utils/MTA_Net.py`.

---

# 中文说明

本项目提供MTA-Net模型、HMS-LBP特征、纹理调制、CBAM、训练与测试入口，以及χ²距离、MAE、NRMSE评价函数。

仓库包含 **202张RGB TIFF图像和202行五粒级实测标签**。图像位于 `samples/images/`，编号为 `wh1-001.tif` 至 `wh1-202.tif`。五个标签列依次对应 <100、100–250、250–500、500–1000和1000–2000 μm，单位为体积百分数。加载后按行归一化为体积分数，CSV测量值保持原样。

主实验划分为145/26/31，区段分组划分为143/20/39。两套划分均完整覆盖202个样本，各子集互不重叠。

## 快速使用

论文训练环境使用Python 3.11。创建独立环境并根据CPU/CUDA安装匹配的PyTorch与torchvision，安装 `requirements.txt`。在项目根目录执行：

```bash
python verify_dataset.py
python train_MTA_Net.py --seed 21 --output results/seed21
python test_MTA_Net.py --checkpoint results/seed21/best_checkpoint.pt --output results/test_seed21
```

训练主干使用ImageNet初始化，首次运行时下载或读取本地缓存的初始化权重；训练产生任务权重 `best_checkpoint.pt`，供测试脚本使用。每个种子使用独立输出目录，五种子循环命令见英文部分。模型种子改变不会改变固定的数据集归属。

## 方法设置

RGB分支保持宽高比缩放并灰色填充至224×224，使用ImageNet归一化参数，训练时施加亮度、对比度和饱和度扰动。HMS分支先由原RGB计算灰度再缩放，统计时排除填充相关码。半径{1,2}取平均，59类uniform-u2映射与4×4网格产生 **944维** 描述符。

纹理特征在Stage2后注入，CBAM在Stage4后应用。训练采用AdamW、batch size 8、50轮、学习率0.0005，按验证χ²选取最佳权重，验证MAE用于并列比较。损失权重、调度器和其他设置见上方配置表及 `Parameters.py`。

普通ResNet34及所有ResNet34消融配置共用MTA-Net的512 → 512 → 256 → 5预测头，BatchNorm、ReLU和Dropout（0.3/0.2）设置相同。比较、消融和分组验证使用同一ResNet34基准实现。

## 文件与输出

数据读取位于 `Datasets/read_labels.py`，批处理位于 `Datasets/data_loader.py`，模型位于 `Utils/MTA_Net.py`，损失、训练与评价指标位于 `Utils/Network_functions.py`。完整文件功能见上方表格。

训练和测试输出逐样本预测、整体及分区间指标，训练另存最佳权重和逐轮记录。NRMSE按当前评价集各粒级的实测极差归一化，再对五个区间等权平均。五种子汇总使用均值和样本标准差。

区段分组命令及其他实验组件说明见英文部分和 [Experiments/README.md](Experiments/README.md)，数据说明见 [samples/README.md](samples/README.md)。

## Automated Tests

```bash
python -m unittest discover -s tests -v
```

The test suite covers data integrity, both splits, all neural experiment forward passes, shared prediction heads, padding masks, colour/texture separation, metric formulas, backpropagation and checkpoint loading. All 18 tests passed with PyTorch 2.8.0 CPU and torchvision 0.23.0.
