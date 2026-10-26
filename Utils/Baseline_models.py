
import torch
import torch.nn as nn
from torchvision import models as tv_models
from Utils.MTA_Net import build_prediction_head

class BaseUnifiedModel(nn.Module):
    def get_model_info(self):
        return {
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'backbone_name': self.__class__.__name__.lower()
        }

class SimpleCNNRegressor(BaseUnifiedModel):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, lbp_features=None):
        x = self.features(x)
        x = self.classifier(x)
        return x

class TorchvisionBackboneRegressor(BaseUnifiedModel):
    def __init__(self, backbone_name='resnet34', num_classes=5, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == 'resnet34':
            self.backbone = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT if pretrained else None)
            self.backbone.fc = build_prediction_head(num_classes)
        elif backbone_name == 'resnet50':
            self.backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_dim, num_classes)
        elif backbone_name == 'efficientnet':
            self.backbone = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            in_dim = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_dim, num_classes)
        else:
            raise ValueError(f"不支持的 backbone_name: {backbone_name}")

    def forward(self, x, lbp_features=None):
        return self.backbone(x)

    def get_model_info(self):
        info = super().get_model_info()
        info['backbone_name'] = self.backbone_name
        return info

def initialize_baseline(name, pretrained=True):
    if name == 'cnn':
        return SimpleCNNRegressor(num_classes=5)
    if name not in ('resnet34','resnet50','efficientnet'):
        raise ValueError('Available components: cnn, resnet34, resnet50, efficientnet')
    return TorchvisionBackboneRegressor(name,num_classes=5,pretrained=pretrained)
