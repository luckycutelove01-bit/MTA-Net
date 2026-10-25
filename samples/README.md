# Dataset

The dataset contains 202 RGB TIFF images and 202 measured grain-size composition vectors.

| File | Description |
|---|---|
| `images/wh1-001.tif` … `images/wh1-202.tif` | One image per sample |
| `labels.csv` | Five measured volume percentages per `Sample_ID` |
| `sample_manifest.csv` | Image SHA256 and main fixed-split membership |
| `DATASET_VERIFICATION.json` | Dataset integrity checks |

Label columns are `Sample_ID,lt100,100_250,250_500,500_1000,1000_2000`, corresponding to <100, 100–250, 250–500, 500–1000 and 1000–2000 μm. IDs include the `.tif` extension. The measured row totals range from 99.9998% to 100.0002% due to rounding. The loader normalizes rows to fractions summing to one.

The main split is `../Datasets/fixed_split_seed21.json` (145/26/31). The profile split is `../Experiments/profile_split_C.json` (143/20/39). `Original_Subset` in the sample manifest refers to the main split.

From the repository root, run `python verify_dataset.py` to decode all images and check IDs, hashes, labels and split coverage.

## 中文

本目录公开202张RGB TIFF图像和202组实测粒度组成标签。图像与标签通过带 `.tif` 后缀的 `Sample_ID` 一一匹配。五个标签列依次对应 <100、100–250、250–500、500–1000和1000–2000 μm，单位为体积百分数。实测CSV数值保持原样，加载时按行归一化。

`sample_manifest.csv` 的 `Original_Subset` 对应主实验划分，`Image_SHA256` 用于文件校验。两套JSON文件分别记录145/26/31和143/20/39个样本的归属。
