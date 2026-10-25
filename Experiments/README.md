# Manuscript Experiments

All neural configurations use the common data loader, RGB-only colour augmentation, composition loss, optimizer, validation selection and metrics. The dataset membership remains fixed across model seeds.

| Family | Variants | Manuscript section |
|---|---|---|
| comparison | wavelet_rf, glcm_rf, lbp_rf, cnn, resnet34, resnet50, resnet50_lbp, efficientnet, mtanet | 3.4 |
| ablation | resnet34, resnet34_lbp, resnet34_hms, resnet34_cbam, mtanet | 3.5 |
| encoding | standard, rotation_invariant, vertical, isotropic, hms | 3.6 |
| radius | r1, r2, r3, r1_2, r1_3, r2_3, r1_2_3 | 3.7.1 |
| stage | M1, M2, M3, M4 | 3.7.2 |
| profile | resnet34, mtanet | 3.8 |

## Shared Method

- ImageNet initialization is used for the ResNet/EfficientNet backbones; the shallow CNN is randomly initialized.
- The plain ResNet34 reference is shared by comparisons, ablations and profile validation. Module-bearing ResNet34 configurations use the prediction head shown in Figure 2 and select Stage2 texture modulation and/or Stage4 CBAM. The complete ablation has the same parameters and forward operations as MTA-Net.
- LBP encodings share original-luminance preprocessing, a valid-area mask, uniform-pattern mapping and 4×4 local histograms. Standard LBP uses radius 1; horizontal/vertical/isotropic multiscale descriptors average radii {1,2}. Rotation invariance uses the minimum circular rotation of the eight-bit code before uniform lookup, retaining the common 59-bin layout. Neighbourhood/rotation/scale choices are recorded by the extractor configuration.
- The encoding comparison uses Stage2 texture modulation without CBAM; ResNet34+HMS is its primary comparator.
- M1=(2,4), M2=(3,4), M3=(3,3), M4=(2,3), with tuples denoting HMS/CBAM stages. M1 shares the main MTA-Net implementation.
- Profile validation reuses exactly the comparison models, input preparation and training settings. Its default split is 143/20/39.
- Random forests are fitted on the training subset only. LBP+RF uses standard LBP; Wavelet+RF uses three-level average-normalized Haar bands, and GLCM+RF uses its configured statistics. Validation and test samples are evaluated separately.

## Train and Evaluate

Run from the repository root:

```bash
python -m Experiments catalog
python -m Experiments train --family ablation --variant resnet34_hms --seed 21 --output results/ablation_hms/seed21
python -m Experiments evaluate --checkpoint results/ablation_hms/seed21/best_checkpoint.pt --output results/ablation_hms/test21
```

Other configurations:

```bash
python -m Experiments train --family encoding --variant rotation_invariant --seed 21 --output results/encoding_ri/seed21
python -m Experiments train --family radius --variant r1_3 --seed 21 --output results/radius_1_3/seed21
python -m Experiments train --family stage --variant M2 --seed 21 --output results/stage_M2/seed21
python -m Experiments train --family profile --variant resnet34 --seed 21 --output results/profile_resnet34/seed21
python -m Experiments train --family comparison --variant lbp_rf --seed 21 --output results/lbp_rf/seed21
python -m Experiments evaluate --checkpoint results/lbp_rf/seed21/best_checkpoint.joblib --output results/lbp_rf/test21
```

Neural training defaults to 50 epochs, batch size 8 and AdamW at 0.0005. `--epochs`, `--batch-size`, `--device`, `--image-dir`, `--labels` and `--split` can be supplied. RF uses `.joblib` checkpoints; neural models use `.pt`. Checkpoints store the selected experiment and texture settings so evaluation reconstructs the matching model and descriptor.

## Five Seeds

PowerShell:

```powershell
foreach ($modelSeed in 21,42,63,84,105) {
    python -m Experiments train --family ablation --variant resnet34_hms --seed $modelSeed --output "results/ablation_hms/seed$modelSeed"
    if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $modelSeed" }
}
python -m Experiments summarize --root results/ablation_hms --output results/ablation_hms/summary.json
```

The summarizer checks model seeds, identical split memberships and identical experiment settings, then reports mean and sample standard deviation (`ddof=1`) from the five actual run metrics.

## Files

| File | Function |
|---|---|
| runner.py / __main__.py | Train, evaluate, list configurations and summarize seeds |
| factory.py | Construct the selected model and texture extractor |
| experiment_catalog.json | Configuration variants and manuscript section mapping |
| comparison_models.py | Shared-stage ablations and ResNet50 with standard LBP |
| texture_encodings.py | Standard, rotation-invariant, vertical and isotropic encodings |
| stage_models.py | M1–M4 models built on the main implementation |
| profile_models.py | Main-comparison models for profile validation |
| profile_split_C.json | Fixed profile split sample IDs |
| rf_models.py | Random forests and handcrafted descriptors |
| statistics.py | Validated five-seed mean/standard deviation |

## 中文说明

所有神经网络实验共用数据加载、RGB颜色增强、组合损失、优化器、验证选模和评价指标。对比、消融和分组验证共用同一ResNet34基准；带模块的ResNet34配置使用图2所示预测头，完整消融模型与主模型一致。编码实验在无CBAM的同一纹理调制模型中比较；半径与阶段实验只改变相应配置；分组验证复用主实验模型，默认使用143/20/39划分。随机森林仅在训练子集拟合，LBP+RF使用标准LBP。

以上命令涵盖训练、权重测试和五种子汇总。各配置的输出均由实际运行产生，模型和纹理设置写入权重及JSON记录。
