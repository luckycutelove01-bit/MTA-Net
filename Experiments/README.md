# 论文其他实验代码整理

本目录整理现有源码中的模型、特征算法、实验配置及来源。没有图像/标签文件读取代码，没有训练好的权重，也没有启动训练或生成论文结果。`python -m Experiments`只列出配置，不下载权重、不创建模型、不训练。

## 实验覆盖

| 论文实验 | 已纳入内容 | 与当前稿件的对应状态 |
| --- | --- | --- |
| 3.4 / Table3 九模型对比 | 当前MTA-Net、CNN、预训练ResNet34/50/EfficientNet-B0；历史ResNet50-LBP；GLCM、Haar-like wavelet、LBP特征和RF构造 | 模型/算法组件齐备；历史部分存在版本差异，不是完整九模型复现入口 |
| 3.5 / Table4 五组消融 | 旧ResNet34、标准LBP、HMS、CBAM、完整旧模型的实际构造函数 | 保留历史实现，不能冒称修订版等条件消融 |
| 3.6 / Table5 编码比较 | 标准、旋转不变、垂向、各向同性多尺度、HMS五个旧提取器；无CBAM旧ResNet34模型 | 编码源码完整；修订版mask/uniform59对照尚未确认 |
| 3.7.1 / Table6 邻域半径 | 七组{1}/{2}/{3}/{1,2}/{1,3}/{2,3}/{1,2,3}，使用当前mask提取器及当前网络构造 | 能构造这些组件；不代表旧结果由此实现产生，未重跑 |
| 3.7.2 / Table7 注入位置 | 已有PaperMTANet的M1(2,4)、M2(3,4)、M3(3,3)、M4(2,3) | 与修订稿阶段组合对应；完整数据加载及训练仍缺失 |
| 3.8 / Tables8–9 剖面分组 | 原143/20/39固定清单、历史兼容MTA-Net、对应纯ResNet34类 | 已记录的运行明确mask=False，不能直接当作当前mask版本验证 |

MTA-Net在编码比较中的参考结果可通过comparison/mtanet组件表示，但不是五个旧编码模型在当前修订网络上的等条件实验。

## 文件说明

- `experiment_catalog.json`：固定种子、六类实验配置、版本状态和限制。配置条目不是已完成运行结果。
- `factory.py`：`build_components(family, variant, seed=21, allow_historical=False)`，返回模型、纹理提取器（或None）、来源说明。构造可能下载ImageNet初始化权重，不执行训练。
- `legacy_lbp.py`：从model.py提取原有五类编码算法，不重新设计映射。
- `legacy_models.py`：原有块内融合、消融和ResNet50-LBP类，不含VGG或ResNet18。
- `stage_models.py`：从当前paper_model.py提取完整M1–M4模型类；主模型Utils/MTA_Net.py仍保留默认M1限制。
- `rf_models.py`：GLCM及三层Haar-like特征、历史预测归一化和RF构造。只接收内存数组。
- `profile_models.py`：原方案C的历史模块命名兼容类和去除纹理/CBAM的控制基线。模型结构与输入特征版本是不同问题。
- `profile_split_C.json`：直接复制原记录，不重新划分。
- `SOURCE_MANIFEST.json`：原始文件路径、SHA256及提取符号/行号。
- `statistics.py`：对调用者提供的五次整体指标计算mean与样本SD（ddof=1），缺失或重复seed报错；不读取图像或标签。
- `requirements-experiments.txt`：RF组件额外依赖scikit-learn；版本未从论文指定，不伪造锁定版本。

## 必须保留的版本差异

1. **旧LBP不是当前HMS的uniform59-mask实现。** 旧代码逐图min–max后转uint8，使用`code % 59`，边缘初值也进入块直方图；没有几何mask。当前HMS使用59类uniform映射、排除中心/全部邻点无效编码。不得互换后仍引用原结果。
2. **旧消融融合位置和公式不同。** 旧模型在Stage2多个BasicBlock内执行加性`F+alpha*M`，CBAM也在块内；当前修订版是在Stage2后一次`F+gamma*F*M`、Stage4后一次CBAM。旧主干随机初始化。交付中三个正式基线组件仍使用用户已要求的ImageNet预训练，历史模块另行标注，不冒充正式修订组件。
3. **RF原始拟合用了训练+验证。** `train_rf_model_one_seed`把145与26合并为171样本；论文表述验证集用于选择。这里仅保留估计器，不自动决定新的拟合样本，不能默默改成145样本后声称复现旧结果。500棵树、max_features=sqrt、min_samples_leaf=2、random_state=seed、MultiOutputRegressor来自原实现。
4. **名称与实际特征需要核实。** 原LBP+RF的默认配置取HMS；已保存ResNet50-LBP Seed21配置也写lbp_type=hms。这与“标准LBP”不是同一描述。Wavelet实现为三层手写Haar-like分解统计，不引入另一个小波库替换。
5. **方案C实际是无掩膜历史实验。** 原manifest写mask=false，run_scheme_c读取historical_unmasked_features.pt并在checkpoint记录mask=False。现在未附缓存或权重，也不恢复其加载函数。用当前mask特征重做属于新实验，不能作为原方案C结果直接引用。

## 使用边界

查看目录：

```bash
python -m Experiments
```

在自行提供内存数据、训练接口和依赖后可构造组件，例如：

```python
from Experiments.factory import build_components
from Utils.Network_functions import set_seed

set_seed(21)
model, texture, spec = build_components('stage', 'M2')
# 后续自行提供RGB、对应HMS特征、实测标签及样本ID。
```

历史组件需显式`allow_historical=True`，默认拒绝将其当作修订版使用。RF模型通过返回对象的fit/predict接口使用；应使用原有normalize_distribution后评价。未调用fit，不生成任何RF权重。

当前`train_MTA_Net.py`及`test_MTA_Net.py`仍因数据加载缺失而退出；其中train_model固定构造主MTA-Net，不是通用实验训练器，不能直接用它替换训练M2或基线。新组件的模型构造、特征接口已组织好，用户仍需另写数据加载及相应训练接线。不会提供看似能运行却实际无加载实现的实验启动器。

所有指标复用`Utils/Network_functions.py:distribution_metrics`；不能混合五seed预测再评价。五seed统计应基于五次独立整体指标，原统计函数使用ddof=1；本文档不填入或生成任何指标数值。

Grad-CAM、绘图、分辨率/颜色敏感性及跨数据集验证未新增。手工更新的Origin绘图CSV和模拟数据未收录。
