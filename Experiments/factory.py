
from pathlib import Path
import json
import warnings

def catalog():
    return json.loads(Path(__file__).with_name('experiment_catalog.json').read_text(encoding='utf-8'))

def build_components(family, variant, seed=21, include_alternative_implementations=False):
    """Return (model, texture_extractor_or_None, configuration).

    No training is performed. Set the training RNG before calling this function.
    RF receives the supplied seed in its estimator configuration.
    """
    spec=dict(catalog()[family][variant])
    if spec.get('implementation') == 'alternative':
        if not include_alternative_implementations:
            raise ValueError('Select include_alternative_implementations=True for this configuration: '+spec['reason'])
        warnings.warn(spec['reason'],RuntimeWarning)
    from Experiments import comparison_models as lm, texture_encodings as ll
    from Utils.HMS_LBP import UniformHMSLBPFeatureExtractor
    from Utils.MTA_Net import MTA_Net
    from Utils.Baseline_models import initialize_baseline
    from Experiments.stage_models import PaperMTANet
    if family=='comparison':
        if variant.endswith('_rf'):
            from Experiments.rf_models import build_rf
            return build_rf(seed),None,spec
        if variant=='mtanet':return MTA_Net(),UniformHMSLBPFeatureExtractor(),spec
        if variant=='resnet50_lbp':return lm.ResNet50WithLBP(),ll.LBPFeatureExtractor(),spec
        return initialize_baseline(variant),None,spec
    if family=='ablation':
        if variant=='resnet34':return lm.TorchvisionBackboneRegressor('resnet34'),None,spec
        if variant=='resnet34_cbam':return lm.resnet34_cbam_only(),None,spec
        if variant=='mtanet':return lm.resnet34_lbp_integrated(),ll.LBPFeatureExtractor(),spec
        extractor=ll.TraditionalLBPFeatureExtractor() if variant=='resnet34_lbp' else ll.LBPFeatureExtractor()
        return lm.resnet34_lbp_only(),extractor,spec
    if family=='encoding':
        extractors={'standard':ll.TraditionalLBPFeatureExtractor,'rotation_invariant':ll.RotationInvariantLBPFeatureExtractor,'vertical':ll.VerticalLBPFeatureExtractor,'isotropic':ll.IsotropicMultiScaleLBPFeatureExtractor,'hms':ll.LBPFeatureExtractor}
        return lm.resnet34_lbp_only(),extractors[variant](),spec
    if family=='radius':return PaperMTANet(),UniformHMSLBPFeatureExtractor(radii=spec['radii']),spec
    if family=='stage':return PaperMTANet(hms_stage=spec['hms_stage'],cbam_stage=spec['cbam_stage']),UniformHMSLBPFeatureExtractor(),spec
    if family=='profile':
        from Experiments.profile_models import ProfileMTANet,PaperResNet34Baseline
        model=ProfileMTANet(True) if variant=='mtanet' else PaperResNet34Baseline(True)
        return model,None,spec
    raise ValueError(family)
