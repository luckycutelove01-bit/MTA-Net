
import torch
from torch import nn
from torchvision import models as tv_models

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        hidden = max(in_channels // reduction_ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class LBPFusionModule(nn.Module):
    """
    保守增强版：
    1) LayerNorm 稳定 LBP 特征幅值；
    2) 零初始化末层，初始几乎等于原 backbone；
    3) 残差式调制，避免纯抑制过强。
    """
    def __init__(self, channel_dim, lbp_feature_dim, reduction_ratio=4, init_scale=0.10):
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

    def forward(self, cnn_features, lbp_features):
        b, c, _, _ = cnn_features.shape
        z = self.norm(lbp_features)
        z = self.fc1(z)
        z = self.bn1(z)
        z = self.act(z)
        z = self.fc2(z).view(b, c, 1, 1)
        return cnn_features + self.alpha * z


class LBPEnhancedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 use_cbam=False, use_lbp_fusion=False, lbp_feature_dim=0):
        super().__init__()
        self.use_cbam = use_cbam
        self.use_lbp_fusion = use_lbp_fusion

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

        if self.use_lbp_fusion and lbp_feature_dim > 0:
            self.lbp_fusion = LBPFusionModule(out_channel, lbp_feature_dim)
        if self.use_cbam:
            self.cbam = CBAM(out_channel)

    def forward(self, x, lbp_features=None):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_lbp_fusion and lbp_features is not None:
            out = self.lbp_fusion(out, lbp_features)
        if self.use_cbam:
            out = self.cbam(out)

        out += identity
        out = self.relu(out)
        return out


class ResNetWithLBPIntegration(nn.Module):
    def __init__(self, block, blocks_num, num_classes=5,
                 cbam_start_layer=4, lbp_integration_layers=[2],
                 lbp_grid_size=4, backbone_name='resnet'):
        super().__init__()
        self.in_channel = 64
        self.lbp_integration_layers = lbp_integration_layers if lbp_integration_layers is not None else []
        self.cbam_start_layer = cbam_start_layer
        self.use_lbp = len(self.lbp_integration_layers) > 0
        self.lbp_feature_dim = 59 * lbp_grid_size * lbp_grid_size if self.use_lbp else 0
        self.backbone_name = backbone_name

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  blocks_num[0], stride=1, layer_num=1)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2, layer_num=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2, layer_num=3)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2, layer_num=4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1, layer_num=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        use_lbp_fusion = layer_num in self.lbp_integration_layers
        use_cbam = self.cbam_start_layer is not None and layer_num >= self.cbam_start_layer

        layers.append(
            block(
                self.in_channel, channel, stride=stride, downsample=downsample,
                use_cbam=use_cbam,
                use_lbp_fusion=use_lbp_fusion,
                lbp_feature_dim=self.lbp_feature_dim if use_lbp_fusion else 0
            )
        )
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(
                    self.in_channel, channel,
                    use_cbam=use_cbam,
                    use_lbp_fusion=use_lbp_fusion,
                    lbp_feature_dim=self.lbp_feature_dim if use_lbp_fusion else 0
                )
            )
        return nn.ModuleList(layers)

    def _forward_layer(self, layer, x, lbp_features):
        for block in layer:
            if getattr(block, 'use_lbp_fusion', False):
                x = block(x, lbp_features)
            else:
                x = block(x)
        return x

    def forward(self, x, lbp_features=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self._forward_layer(self.layer1, x, lbp_features)
        x = self._forward_layer(self.layer2, x, lbp_features)
        x = self._forward_layer(self.layer3, x, lbp_features)
        x = self._forward_layer(self.layer4, x, lbp_features)

        x = self.avgpool(x)
        cnn_features = torch.flatten(x, 1)
        logits = self.classifier(cnn_features)
        return logits

    def get_model_info(self):
        info = {
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'lbp_feature_dim': self.lbp_feature_dim if self.use_lbp else 0,
            'lbp_integration_layers': self.lbp_integration_layers,
            'cbam_start_layer': self.cbam_start_layer,
            'backbone_name': self.backbone_name
        }
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], 1):
            info[f'layer{i}_lbp'] = any(getattr(b, 'use_lbp_fusion', False) for b in layer)
            info[f'layer{i}_cbam'] = any(getattr(b, 'use_cbam', False) for b in layer)
        return info


def resnet34_lbp_integrated(num_classes=5, cbam_start_layer=4,
                            lbp_integration_layers=[2], lbp_grid_size=4):
    return ResNetWithLBPIntegration(
        LBPEnhancedBasicBlock, [3, 4, 6, 3],
        num_classes=num_classes,
        cbam_start_layer=cbam_start_layer,
        lbp_integration_layers=lbp_integration_layers,
        lbp_grid_size=lbp_grid_size,
        backbone_name='resnet34_lbp_cbam'
    )


def resnet34_lbp_only(num_classes=5, lbp_integration_layers=[2], lbp_grid_size=4):
    return ResNetWithLBPIntegration(
        LBPEnhancedBasicBlock, [3, 4, 6, 3],
        num_classes=num_classes,
        cbam_start_layer=None,
        lbp_integration_layers=lbp_integration_layers,
        lbp_grid_size=lbp_grid_size,
        backbone_name='resnet34_lbp'
    )


def resnet34_cbam_only(num_classes=5, cbam_start_layer=4, lbp_grid_size=4):
    return ResNetWithLBPIntegration(
        LBPEnhancedBasicBlock, [3, 4, 6, 3],
        num_classes=num_classes,
        cbam_start_layer=cbam_start_layer,
        lbp_integration_layers=[],
        lbp_grid_size=lbp_grid_size,
        backbone_name='resnet34_cbam'
    )


class BaseUnifiedModel(nn.Module):
    def get_model_info(self):
        return {
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'backbone_name': self.__class__.__name__.lower()
        }


class TorchvisionBackboneRegressor(BaseUnifiedModel):
    def __init__(self, backbone_name='resnet34', num_classes=5):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == 'vgg':
            self.backbone = tv_models.vgg16(weights=None)
            in_dim = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_dim, num_classes)
        elif backbone_name == 'resnet34':
            self.backbone = tv_models.resnet34(weights=None)
            in_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_dim, num_classes)
        elif backbone_name == 'resnet50':
            self.backbone = tv_models.resnet50(weights=None)
            in_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_dim, num_classes)
        elif backbone_name == 'efficientnet':
            self.backbone = tv_models.efficientnet_b0(weights=None)
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


class ResNet50WithLBP(BaseUnifiedModel):
    def __init__(self, num_classes=5, lbp_grid_size=4):
        super().__init__()
        self.lbp_feature_dim = 59 * lbp_grid_size * lbp_grid_size
        backbone = tv_models.resnet50(weights=None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.lbp_fusion = LBPFusionModule(channel_dim=512, lbp_feature_dim=self.lbp_feature_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, lbp_features=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if lbp_features is not None:
            x = self.lbp_fusion(x, lbp_features)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_model_info(self):
        info = super().get_model_info()
        info['lbp_feature_dim'] = self.lbp_feature_dim
        info['lbp_integration_layers'] = [2]
        info['cbam_start_layer'] = None
        info['backbone_name'] = 'resnet50_lbp_layer2'
        return info

