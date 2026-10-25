# Experiment Components

This directory exposes component constructors and configuration metadata. `python -m Experiments` prints the catalog. Main MTA-Net training/evaluation commands are in the root README.

| File | Function |
|---|---|
| `experiment_catalog.json` | Seeds, component names and configuration descriptions |
| `factory.py` | Construct a model and its texture extractor |
| `comparison_models.py` | Residual-block fusion, ablation constructors and ResNet50 texture fusion |
| `texture_encodings.py` | Horizontal, standard, rotation-invariant, vertical and isotropic encoders |
| `stage_models.py` | Stage-level MTA-Net for M1–M4 insertion stages |
| `profile_models.py` | Profile model parameter naming and ResNet34 baseline |
| `profile_split_C.json` | Fixed 143/20/39 sample memberships |
| `rf_models.py` | Random forests and wavelet-like, GLCM and horizontal texture features |
| `statistics.py` | Five-run mean and sample standard deviation |
| `SOURCE_MANIFEST.json` | Component source hashes |

## Construct Components

```python
from Utils.Network_functions import set_seed
from Experiments.factory import build_components

set_seed(21)
model, extractor, configuration = build_components('stage', 'M1', seed=21)
```

M1=(2,4), M2=(3,4), M3=(3,3), M4=(2,3); pairs denote HMS/CBAM stages. Radius variants select subsets of {1,2,3} and average the descriptors.

The catalog distinguishes stage-level MTA-Net components (`implementation: main`) from alternative designs (`implementation: alternative`). Explicitly select alternative designs:

```python
model, extractor, configuration = build_components(
    'encoding', 'standard', seed=21,
    include_alternative_implementations=True,
)
```

## Configuration Details

`Utils/HMS_LBP.py` uses geometric valid-area masks and a uniform-u2 lookup for 59-bin histograms. `texture_encodings.py` uses modulo-59 assignment, resized grayscale inputs and spatial histograms. These are different preprocessing and binning rules.

`Utils/MTA_Net.py` and `stage_models.py` use ImageNet initialization and stage-level residual channel modulation. `comparison_models.py` includes randomly initialized residual-block fusion networks and a randomly initialized ResNet50 texture model. Comparison torchvision baselines in `Utils/Baseline_models.py` use ImageNet initialization.

The factory constructs components; its `seed` configures random forests. Set the RNG before constructing neural models. Callers supply loading, training, selection and outputs when integrating other components into experiment workflows. Root `train_MTA_Net.py` and `test_MTA_Net.py` instantiate the main MTA-Net.

The profile split can be passed directly to root MTA-Net scripts. `profile_models.py` also exposes a ResNet34 baseline and a model using `texture_projection`/`classifier` parameter names.

## Statistics

`summarize_five_runs(rows)` takes five dictionaries with `seed`, `chi_square`, `mae`, `nrmse` (percent). It validates seeds 21/42/63/84/105 and reports mean and sample standard deviation (`ddof=1`). Supply actual run-level metrics.

## 中文说明

本目录提供其他模型、纹理编码、半径和注入阶段的组件及配置，以及五种子统计函数。主模型使用uniform-u2映射、有效区域掩膜、ImageNet初始化和Stage2纹理调制；部分组件使用模59分箱、随机初始化或残差块级融合。各文件与配置清单说明了这些不同的计算过程。

主目录训练入口构造MTA-Net；其他组件由factory接口构造并接入相应训练流程。143/20/39区段划分可以通过主入口的 `--split` 参数使用。
