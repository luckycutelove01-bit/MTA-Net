"""Shared-stage ResNet34 ablations and a ResNet50 texture baseline."""
import torch
from torch import nn
from torchvision import models as tv_models

from Utils.MTA_Net import MTA_Net


class ResNet34Ablation(MTA_Net):
    """Keep the backbone/head and switch only texture modulation and CBAM."""
    def __init__(self, use_texture=True, use_cbam=True, pretrained=True):
        super().__init__(pretrained=pretrained)
        self.use_texture = bool(use_texture)
        self.use_cbam = bool(use_cbam)
        if not self.use_texture:
            del self.hms_projection
        if not self.use_cbam:
            del self.cbam

    def _after_stage(self, stage, feature, hms_features):
        if self.use_texture and self.hms_stage == stage:
            feature = self.hms_projection(feature, hms_features)
        if self.use_cbam and self.cbam_stage == stage:
            feature = self.cbam(feature)
        return feature

    def get_model_info(self):
        return {'backbone': 'ResNet34', 'use_texture': self.use_texture,
                'use_cbam': self.use_cbam, 'hms_stage': self.hms_stage,
                'cbam_stage': self.cbam_stage,
                'params_m': sum(p.numel() for p in self.parameters()) / 1e6}


class LBPFusionModule(nn.Module):
    """Normalize and project LBP to an additive Stage2 residual."""
    def __init__(self, channel_dim=512, lbp_feature_dim=944,
                 reduction_ratio=4, init_scale=0.10):
        super().__init__()
        hidden_dim = max(channel_dim // reduction_ratio, 16)
        self.norm = nn.LayerNorm(lbp_feature_dim)
        self.fc1 = nn.Linear(lbp_feature_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, channel_dim)
        self.alpha = nn.Parameter(torch.tensor(float(init_scale)))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feature, texture):
        b, c, _, _ = feature.shape
        z = self.norm(texture)
        z = self.act(self.bn1(self.fc1(z)))
        z = self.fc2(z).view(b, c, 1, 1)
        return feature + self.alpha * z


class ResNet50WithLBP(nn.Module):
    """ImageNet-initialized ResNet50 with standard LBP fusion after Stage2."""
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.DEFAULT if pretrained else None)
        for name in ('conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2',
                     'layer3', 'layer4', 'avgpool'):
            setattr(self, name, getattr(backbone, name))
        self.lbp_fusion = LBPFusionModule()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(2048, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(512, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, num_classes),
        )

    def forward(self, image, lbp_features):
        x = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        x = self.layer1(x)
        x = self.lbp_fusion(self.layer2(x), lbp_features)
        x = self.layer4(self.layer3(x))
        return self.classifier(torch.flatten(self.avgpool(x), 1))
