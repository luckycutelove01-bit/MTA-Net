"""Scheme-C network classes only. Historical runs used unmasked cached features.
Do not substitute current masked features and call that the old experiment.
"""
import torch
from Experiments import stage_models as pm

class HistoricalCompatiblePaper(pm.PaperMTANet):
    """Current forward structure with historical module names, validated by checkpoint inference.
    Renaming preserves registration order and numerical operations. No extra learned parameters.
    """
    def __init__(self,pretrained=False):
        super().__init__(pretrained=pretrained)
        self._modules={({'hms_projection':'texture_projection','prediction_head':'classifier'}.get(k,k)):v for k,v in self._modules.items()}
    def _after_stage(self,stage,feature,hms_features):
        if self.hms_stage==stage:feature=self.texture_projection(feature,hms_features)
        if self.cbam_stage==stage:feature=self.cbam(feature)
        return feature
    def forward(self,image,hms_features):
        x=self.maxpool(self.relu(self.bn1(self.conv1(image))))
        x=self._after_stage(1,self.layer1(x),hms_features)
        x=self._after_stage(2,self.layer2(x),hms_features)
        x=self._after_stage(3,self.layer3(x),hms_features)
        x=self._after_stage(4,self.layer4(x),hms_features)
        return self.classifier(torch.flatten(self.avgpool(x),1))


class PaperResNet34Baseline(HistoricalCompatiblePaper):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)
        del self.texture_projection
        del self.cbam

    def _after_stage(self, stage, feature, hms_features):
        return feature

