"""Insertion-stage variants sharing the MTA-Net implementation."""
from Utils.MTA_Net import MTA_Net


class PaperMTANet(MTA_Net):
    def __init__(self, num_classes=5, hms_stage=2, cbam_stage=4, pretrained=True):
        if hms_stage not in (2, 3) or cbam_stage not in (3, 4):
            raise ValueError('Expected HMS stage 2/3 and CBAM stage 3/4')
        super().__init__(num_classes=num_classes, hms_stage=hms_stage,
                         cbam_stage=cbam_stage, pretrained=pretrained)
