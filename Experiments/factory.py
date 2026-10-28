"""Construct experiment models with shared preprocessing and training conventions."""
from pathlib import Path
import json


def catalog():
    return json.loads(Path(__file__).with_name('experiment_catalog.json').read_text(encoding='utf-8'))


def build_components(family, variant, seed=21, pretrained=True):
    """Return model, texture extractor (or None), and the selected configuration."""
    spec = dict(catalog()[family][variant])
    from Utils.HMS_LBP import UniformHMSLBPFeatureExtractor
    from Utils.MTA_Net import MTA_Net
    from Utils.Baseline_models import initialize_baseline
    from Experiments.comparison_models import ResNet34Ablation, ResNet50WithLBP
    from Experiments.stage_models import PaperMTANet
    from Experiments.texture_encodings import (
        TraditionalLBPFeatureExtractor, RotationInvariantLBPFeatureExtractor,
        VerticalLBPFeatureExtractor, IsotropicMultiScaleLBPFeatureExtractor,
    )
    if family in ('comparison', 'profile'):
        if variant.endswith('_rf'):
            from Experiments.rf_models import build_rf
            return build_rf(seed), None, spec
        if variant == 'mtanet':
            return MTA_Net(pretrained=pretrained), UniformHMSLBPFeatureExtractor(), spec
        if variant == 'resnet50_lbp':
            return ResNet50WithLBP(pretrained=pretrained), TraditionalLBPFeatureExtractor(), spec
        return initialize_baseline(variant, pretrained=pretrained), None, spec
    if family == 'ablation':
        if variant == 'resnet34':
            return initialize_baseline('resnet34', pretrained=pretrained), None, spec
        use_texture = variant in ('resnet34_lbp', 'resnet34_hms', 'mtanet')
        use_cbam = variant in ('resnet34_cbam', 'mtanet')
        model = ResNet34Ablation(use_texture, use_cbam, pretrained=pretrained)
        texture = (TraditionalLBPFeatureExtractor() if variant == 'resnet34_lbp'
                   else UniformHMSLBPFeatureExtractor()) if use_texture else None
        return model, texture, spec
    if family == 'encoding':
        extractors = {'standard': TraditionalLBPFeatureExtractor,
                      'rotation_invariant': RotationInvariantLBPFeatureExtractor,
                      'vertical': VerticalLBPFeatureExtractor,
                      'isotropic': IsotropicMultiScaleLBPFeatureExtractor,
                      'hms': UniformHMSLBPFeatureExtractor}
        return ResNet34Ablation(True, False, pretrained=pretrained), extractors[variant](), spec
    if family == 'radius':
        return MTA_Net(pretrained=pretrained), UniformHMSLBPFeatureExtractor(radii=spec['radii']), spec
    if family == 'stage':
        return (PaperMTANet(hms_stage=spec['hms_stage'], cbam_stage=spec['cbam_stage'],
                            pretrained=pretrained), UniformHMSLBPFeatureExtractor(), spec)
    raise ValueError(f'Unknown experiment family: {family}')
