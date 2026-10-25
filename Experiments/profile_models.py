"""Profile-segment evaluation uses the same models as the main comparison."""
from Utils.MTA_Net import MTA_Net
from Utils.Baseline_models import TorchvisionBackboneRegressor


class ProfileMTANet(MTA_Net):
    pass


class PaperResNet34Baseline(TorchvisionBackboneRegressor):
    def __init__(self, pretrained=True):
        super().__init__('resnet34', pretrained=pretrained)
