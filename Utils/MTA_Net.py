import torch
import torch.nn as nn
from torchvision import models as tv_models
from Parameters import HMS_FEATURE_DIM,HMS_RADII,HMS_STAGE,CBAM_STAGE,CBAM_REDUCTION,DROPOUT,NUM_CLASSES
from Utils.Texture_Injection import HMSProjection
from Utils.CBAM import StageCBAM

class MTA_Net(nn.Module):
    """Revised MTA-Net; Stage2 texture and Stage4 attention, each once."""

    STAGE_CHANNELS = {1: 64, 2: 128, 3: 256, 4: 512}

    def __init__(self, num_classes: int = NUM_CLASSES, hms_stage: int = HMS_STAGE,
                 cbam_stage: int = CBAM_STAGE, pretrained: bool = True):
        super().__init__()
        if hms_stage != HMS_STAGE:
            raise ValueError("This compact project fixes HMS after Stage2")
        if cbam_stage != CBAM_STAGE:
            raise ValueError("This compact project fixes CBAM after Stage4")
        self.hms_stage = int(hms_stage)
        self.cbam_stage = int(cbam_stage)
        weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet34(weights=weights)
        self.backbone_pretrained = bool(pretrained)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.hms_projection = HMSProjection(self.STAGE_CHANNELS[self.hms_stage])
        self.cbam = StageCBAM(self.STAGE_CHANNELS[self.cbam_stage], reduction_ratio=CBAM_REDUCTION)
        self.prediction_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT[0]),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT[1]),
            nn.Linear(256, num_classes),
        )

    def _after_stage(self, stage: int, feature: torch.Tensor,
                     hms_features: torch.Tensor) -> torch.Tensor:
        if self.hms_stage == stage:
            feature = self.hms_projection(feature, hms_features)
        if self.cbam_stage == stage:
            feature = self.cbam(feature)
        return feature

    def forward(self, image: torch.Tensor, hms_features: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        x = self._after_stage(1, self.layer1(x), hms_features)
        x = self._after_stage(2, self.layer2(x), hms_features)
        x = self._after_stage(3, self.layer3(x), hms_features)
        x = self._after_stage(4, self.layer4(x), hms_features)
        return self.prediction_head(torch.flatten(self.avgpool(x), 1))

    def get_model_info(self) -> dict:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "backbone": "ResNet34",
            "backbone_pretrained": "ImageNet" if self.backbone_pretrained else "None",
            "weights": "ResNet34_Weights.DEFAULT" if self.backbone_pretrained else "None",
            "hms_feature_dim": HMS_FEATURE_DIM,
            "hms_radii": list(HMS_RADII),
            "hms_grid": "4x4",
            "hms_bins": 59,
            "hms_stage": self.hms_stage,
            "hms_applications": 1,
            "hms_projection": f"944 -> {self.hms_projection.hidden_dim} -> {self.hms_projection.channels}",
            "gamma_type": "learnable scalar",
            "gamma_initial": 0.10,
            "cbam_stage": self.cbam_stage,
            "cbam_applications": 1,
            "cbam_reduction_ratio": 16,
            "cbam_spatial_kernel": 7,
            "prediction_head": "512 -> 512 -> 256 -> 5",
            "dropout": [0.3, 0.2],
            "output": "5 logits -> Softmax distribution",
            "total_params": total,
            "trainable_params": trainable,
            "params_m": total / 1e6,
        }

