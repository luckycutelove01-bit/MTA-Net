# MTA-Net：精简训练与测试代码

本项目从用户提供的**当前修订版**源码提取，保留ResNet34、HMS-LBP、纹理调制、CBAM、五维组成输出及原复合损失。功能组织参考[Sedihist仓库](https://github.com/weizi8248/Sedihist)的参数、数据、模型、训练、测试和结果保存角色；没有复制其ResNet18、直方图层、九维输出或损失算法。

**当前状态（2026-10-23）：数据加载代码已按要求删除，训练和测试入口不能直接运行。需自行实现加载程序。下述模型检查属于删除前的历史记录，不代表当前交付可端到端运行。** 此交付未运行完整训练，未生成实验预测、指标或任务模型权重，未复现论文精度。源码基础是2026-10-22修订版，不是历史无掩膜Paper版或旧五随机种子模型。

## 其他实验范围更新

`Experiments/`已补充模型对比、消融、LBP编码、邻域半径、注入位置和剖面分组实验的配置与现有模型/特征组件，以及五次整体指标的mean与样本SD汇总函数。详细来源和版本差异见[实验说明](Experiments/README.md)。这些组件不包含数据加载或完整实验训练入口，不能据此声称论文全部实验已复现。

## 文件功能

```text
MTA-Net_Code/
├── Parameters.py                    默认结构、HMS、路径和单次训练参数
├── train_MTA_Net.py                 单模型、单seed训练入口
├── test_MTA_Net.py                  指定权重的带标签固定测试集评价
├── Datasets/
│   ├── __init__.py
│   └── fixed_split_seed21.json      原始145/26/31清单，202个ID
├── Utils/
│   ├── __init__.py
│   ├── MTA_Net.py                   ResNet34整体前向
│   ├── HMS_LBP.py                   缩放填充、mask、完整HMS算法
│   ├── Texture_Injection.py         LN、投影及乘性残差调制
│   ├── CBAM.py                      通道注意力、空间注意力
│   ├── Network_functions.py         初始化、损失、指标、单次训练及评价
│   ├── Save_Results.py              权重、训练记录、逐样本预测、指标保存
│   └── Baseline_models.py           CNN/ResNet34/ResNet50/EfficientNet-B0组件
├── samples/
│   ├── images/                     完整wh1-001.tif…wh1-202.tif
│   ├── labels.csv                  202行原始实测标签（百分数）
│   ├── sample_manifest.csv         原集合归属及图像SHA256
│   └── README.md
├── requirements.txt
├── SOURCE_MANIFEST.json             原源码、标签及划分来源SHA256
├── CHECKS.md                        实际检查结果及未验证范围
└── README.md
```

不包含五seed自动运行、完整比较训练平台、独立无标签预测工具、批量特征缓存、Grad-CAM或绘图工具。新增实验配置及统计函数见`Experiments/`。

## 环境和外部依赖

- 正式训练按当前稿件使用**Python 3.11**；训练CLI保留版本检查。本次仅在Python 3.12.3、PyTorch 2.8.0+cpu、torchvision 0.23.0+cpu环境完成代码检查，尚未验证3.11或GPU运行。
- `requirements.txt`记录本次实际检查的库版本：torch、torchvision、NumPy、pandas、Pillow、openpyxl。其中标签加载实现已删除；RF组件另需安装`Experiments/requirements-experiments.txt`中的scikit-learn，不依赖原项目的本地模块。
- 先按机器环境安装匹配的torch/torchvision CPU或CUDA构建；随后在项目文件夹执行：

```bash
python -m pip install -r requirements.txt
```

库版本是已检查环境的记录，不声称其他Python/CUDA组合已验证。

主模型训练使用`torchvision.models.ResNet34_Weights.DEFAULT`，本次环境对应`IMAGENET1K_V1`。首次使用需要torchvision下载ImageNet权重，或预先将匹配权重放入PyTorch缓存；本包没有附带它。测试加载本项目任务checkpoint时使用`pretrained=False`构建结构，不重新下载ImageNet权重。

## 数据文件与缺失的加载实现

本交付已包含202张原始图像、`samples/labels.csv`和原始202个ID的
`Datasets/fixed_split_seed21.json`清单（145/26/31）。数据文件未删除或改变。

已删除：`Prepare_Data.py`、`Datasets/data_loader.py`、`Datasets/read_labels.py`。
没有图像文件读取、标签CSV/XLSX解析、标签校验归一化或DataLoader组装实现。
模型所需的HMS-LBP算法及对内存图像的缩放、填充和掩膜功能仍保留。

示例CSV列为`Sample_ID,lt100,100_250,250_500,500_1000,1000_2000`，
五值是原始实测百分数，不是已经处理后的模型标签。新加载程序应按既定定义
转换为非负、总和为1的五维组成，保持样本与图像对应。2026-10-25已补齐完整202组数据，核查记录见samples/DATASET_VERIFICATION.json。

## 主模型与数据流

- RGB：保宽高比双线性缩放，居中填充灰色(128,128,128)至224×224；仅训练RGB有brightness/contrast/saturation各0.1颜色扰动；ToTensor及ImageNet均值/标准差归一化。无翻转、旋转或裁剪。
- HMS：原始未扰动RGB按0.299/0.587/0.114转float32灰度，再按相同几何缩放填充；不做ImageNet标准化、不做逐图min–max。
- 半径{1,2}；各8邻点按`(-2r,0),(-r,0),(r,0),(2r,0),(-2r,-r),(2r,-r),(-2r,r),(2r,r)`依次赋2⁰…2⁷；邻点≥中心为1。
- U≤2的58个raw code按整数升序映射0…57，其余映射58。完整画布4×4分块，每块56×56；中心及全部邻点须有效且不越界。每块直方图按有效中心数归一化，空块输出零向量并告警；每尺度944维，尺度间平均，最终944维。
- 在线逐样本提取纹理，没有磁盘特征缓存管理；仅训练RGB颜色扰动，HMS特征确定不变。
- Stage2结束后：`944→LayerNorm→Linear32→BN→ReLU→Linear128`，`F+gamma*F*M`；gamma初值0.10。
- Stage4结束后：CBAM一次，先通道后空间，压缩率16、空间卷积核7。
- GAP512→FC512/BN/ReLU/Dropout0.3→FC256/BN/ReLU/Dropout0.2→FC5。模型`forward`返回logits，损失与评价再Softmax，避免重复Softmax。
- 默认模型参数量21,750,760。没有为匹配论文表格数字改变隐藏维度。

## 损失、训练与评价

每样本基础损失：`0.6*CDF + 0.3*L1 + 0.1*cos`。CDF为5区间累积分布绝对差均值；L1为5区间绝对差均值；cos为`1-cosine_similarity`。真实标签熵计算`w=1+0.6*H/log(5)`，batch损失为`mean(w*base)`，不是除以权重和；epsilon=1e-8。

AdamW：lr=0.0005，weight_decay=1e-4；batch size=8；50epochs；梯度裁剪max_norm=1。ReduceLROnPlateau监测验证损失，factor=.5、patience=8。保留原验证损失的batch等权平均方式（最后小batch仍等权），不擅自改为样本加权均值；它不影响下述全数据指标聚合。

训练索引使用seed+epoch打乱；145个样本的末尾单张batch被丢弃以兼容BatchNorm，每轮18个8张batch，原清单仍为145张。无early stopping。验证χ²优先，1e-12容差内MAE更低者优先，同分保留较早checkpoint。测试使用最佳验证checkpoint。

三项指标均使用0–1组成：

```text
chi2 = mean_i[0.5 * sum_j((pred-true)^2/(pred+true+1e-8))]
MAE = mean_j(mean_i(abs(pred-true)))
RMSE_j = sqrt(mean_i((pred_j-true_j)^2))
NRMSE_j = RMSE_j/(max(true_j)-min(true_j)+1e-8)*100
NRMSE = mean_j(NRMSE_j)
```

极差来自当前评价集的实测值；χ²分区间贡献求和等于总体χ²。主入口每次仅训练和评价一个seed。`Experiments/statistics.py`可汇总五次独立整体指标，不先平均预测。

## 训练与测试入口：需另写加载程序

两个入口保留参数解析和训练/测试调用，但启动时检查本项目的`Prepare_Data.py`。
当前文件不存在，入口会明确退出，且不会读取数据、下载权重或启动训练。
`--help`仍可查看参数。不能将当前交付描述为开箱即用的完整训练项目。

重新启用入口需要自行提供`Prepare_Data.py`及其中的
`prepare_loaders(config, test_only=False)`，或修改入口连接自己的加载程序。
函数需返回`(loaders, split)`：

- `loaders`：训练时包含`train/val/test`；仅测试时包含`test`。
- 每批依次返回`RGB张量、944维HMS张量、五维0–1标签张量、样本ID列表`。
- 张量形状分别为`B×3×224×224`、`B×944`、`B×5`。
- `split`：包含`train/val/test`样本ID列表，使用保留的固定清单，不重新划分。
- 验证与测试无随机增强，颜色增强仅用于训练RGB，HMS仍来自原始灰度分支。
- 若需严格沿用原训练采样，保留seed+epoch打乱及最后单样本batch丢弃规则。

这里只说明接口，不提供替代读取实现或空壳加载函数。训练后的真实权重和指标
保存函数仍在`Utils/`，但必须先补齐数据加载才能使用完整入口。

## 部分对比组件

`Utils/Baseline_models.py`从现有`model.py`保留CNN、ResNet34、ResNet50、EfficientNet-B0及`initialize_baseline(name)`。输出5个logits；使用时再按组成任务Softmax。

**ResNet34、ResNet50、EfficientNet-B0使用ImageNet预训练主干**，分别调用torchvision对应权重枚举的`DEFAULT`；加载主干后替换为新初始化的五维输出层。CNN仍随机初始化。首次构建这些基线需要下载匹配的权重或已有本地缓存；权重文件不随交付包附带。它们仍是组件交付，尚未复现论文基线性能。主模型训练和测试入口始终使用MTA-Net。历史RF特征及估计器、ResNet50-LBP组件已纳入`Experiments/`，版本差异单独说明；未新增VGG或ResNet18模型，也未提供完整比较训练平台。

## 来源与检查边界

算法来源：`paper_model.py`；训练/指标和预处理参数来源：`final_pipeline.py`；基线来源：`model.py`；固定清单来源：`final_run_outputs/fixed_split_seed21.json`。`SOURCE_MANIFEST.json`保存原文件SHA256，整理未修改原文件。保留包内导入、单次入口和结果保存接线；CSV读取实现已按要求删除。其他实验的来源单独记录于`Experiments/SOURCE_MANIFEST.json`。

详见`CHECKS.md`。完整训练、修订版权重推理、GPU和Python3.11环境仍未验证，不据代码检查声称论文精度已复现。


2026-10-23更新：按论文要求将上述三个基线的初始化从随机权重改为ImageNet预训练。此改动仅影响后续模型构建，不证明历史实验使用过相同初始化；旧实验记录与审计报告不追溯改写。
