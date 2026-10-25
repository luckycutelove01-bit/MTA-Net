"""Check shared experiment settings and the invariants stated in the manuscript."""
import gc
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from Experiments.factory import build_components, catalog
from Experiments.runner import experiment_settings, rf_matrix
from Experiments.texture_encodings import (
    TraditionalLBPFeatureExtractor, RotationInvariantLBPFeatureExtractor,
    VerticalLBPFeatureExtractor, IsotropicMultiScaleLBPFeatureExtractor,
)
from Datasets.read_labels import read_records
from Parameters import ROOT, RunParameters
from Prepare_Data import read_split, SafeBatchSampler
from Utils.HMS_LBP import UniformHMSLBPFeatureExtractor, ResizePad224, UNIFORM_U2_LOOKUP
from Utils.MTA_Net import MTA_Net
from Utils.Network_functions import PaperDistributionLoss, evaluate, load_checkpoint
from Utils.Save_Results import save_checkpoint


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_variants_construct_and_forward(self):
        image = torch.randn(2, 3, 224, 224)
        feature = torch.randn(2, 944)
        for family, variants in catalog().items():
            if family == 'seeds':
                continue
            for variant in variants:
                if variant.endswith('_rf'):
                    continue
                with self.subTest(family=family, variant=variant):
                    model, texture, _ = build_components(family, variant, pretrained=False)
                    model.eval()
                    with torch.no_grad():
                        self.assertEqual(tuple(model(image, feature).shape), (2, 5))
                    if texture is not None:
                        self.assertEqual(texture.get_config()['mapping'], 'uniform-u2')
                        self.assertTrue(texture.get_config()['mask'])
                    del model
                    gc.collect()

    def test_complete_ablation_matches_main(self):
        torch.manual_seed(42)
        main = MTA_Net(pretrained=False)
        torch.manual_seed(42)
        ablation, _, _ = build_components('ablation', 'mtanet', pretrained=False)
        self.assertEqual(set(main.state_dict()), set(ablation.state_dict()))
        for key, value in main.state_dict().items():
            torch.testing.assert_close(value, ablation.state_dict()[key], rtol=0, atol=0)

    def test_comparison_ablation_and_profile_share_baseline(self):
        torch.manual_seed(21)
        baseline, _, _ = build_components('comparison', 'resnet34', pretrained=False)
        for family in ('ablation', 'profile'):
            torch.manual_seed(21)
            model, _, _ = build_components(family, 'resnet34', pretrained=False)
            self.assertEqual(set(baseline.state_dict()), set(model.state_dict()))
            for key, value in baseline.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    def test_uniform_mapping_and_padding_for_all_encodings(self):
        image = Image.fromarray(np.full((60, 100, 3), 70, dtype=np.uint8))
        prepared = ResizePad224()(image)
        changed = prepared.padded_gray.copy()
        changed[~prepared.content_mask] = 255
        for cls in (UniformHMSLBPFeatureExtractor, TraditionalLBPFeatureExtractor,
                    RotationInvariantLBPFeatureExtractor, VerticalLBPFeatureExtractor,
                    IsotropicMultiScaleLBPFeatureExtractor):
            with self.subTest(encoder=cls.__name__):
                extractor = cls()
                first, _ = extractor.extract_prepared(prepared.padded_rgb, prepared.content_mask, prepared.padded_gray)
                second, _ = extractor.extract_prepared(prepared.padded_rgb, prepared.content_mask, changed)
                np.testing.assert_array_equal(first, second)
                histogram = first.reshape(16, 59)
                self.assertEqual(np.flatnonzero(histogram.sum(0)).tolist(), [int(UNIFORM_U2_LOOKUP[255])])

    def test_experiment_checkpoint_roundtrip(self):
        model, texture, _ = build_components('encoding', 'standard', pretrained=False)
        model.eval()
        image, features = torch.randn(2, 3, 224, 224), torch.randn(2, 944)
        settings = experiment_settings('encoding', 'standard', texture)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'model.pt'
            save_checkpoint(file, model, 1, {}, RunParameters(),
                            read_split(ROOT / 'Datasets/fixed_split_seed21.json'), settings)
            restored, checkpoint = load_checkpoint(file)
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(image, features), model(image, features), rtol=0, atol=0)
            self.assertEqual(checkpoint['experiment_config']['texture']['encoding'], 'standard')

    def test_rf_lbp_uses_standard_encoder(self):
        records = read_records(ROOT / 'samples/images', ROOT / 'samples/labels.csv')[:1]
        actual = rf_matrix(records, 'lbp_rf')[0]
        with Image.open(records[0].image_path) as image:
            expected, _ = TraditionalLBPFeatureExtractor().extract_pil(image)
        np.testing.assert_array_equal(actual, expected)

    def test_validation_loss_is_sample_weighted(self):
        class Fixed(torch.nn.Module):
            def forward(self, image, texture):
                return image
        logits = torch.tensor([[5., 0, 0, 0, 0], [0., 5, 0, 0, 0], [0., 0, 0, 0, 5]])
        target = torch.tensor([[1., 0, 0, 0, 0], [0., 1, 0, 0, 0], [1., 0, 0, 0, 0]])
        loader = [(logits[:2], torch.zeros(2, 944), target[:2], ['a', 'b']),
                  (logits[2:], torch.zeros(1, 944), target[2:], ['c'])]
        criterion = PaperDistributionLoss()
        loss, *_ = evaluate(Fixed(), loader, criterion, torch.device('cpu'))
        self.assertAlmostEqual(loss, float(criterion(logits, target)), places=6)

    def test_sampler_keeps_all_samples(self):
        for n in (3, 9, 143, 145):
            for size in (2, 3, 8):
                sampler = SafeBatchSampler(range(n), size)
                batches = list(sampler)
                self.assertEqual(len(batches), len(sampler))
                self.assertTrue(all(len(batch) >= 2 for batch in batches))
                self.assertEqual(sorted(i for batch in batches for i in batch), list(range(n)))


if __name__ == '__main__':
    unittest.main()
