import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from Datasets.data_loader import read_image_batch
from Datasets.read_labels import read_records
from Parameters import ROOT, RunParameters
from Prepare_Data import read_split, prepare_loaders, SafeBatchSampler
from Utils.HMS_LBP import ResizePad224, UniformHMSLBPFeatureExtractor, UNIFORM_U2_LOOKUP
from Utils.MTA_Net import MTA_Net
from Utils.Network_functions import PaperDistributionLoss, distribution_metrics, load_checkpoint
from Utils.Save_Results import save_checkpoint
from verify_dataset import verify


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.records = read_records(ROOT / 'samples/images', ROOT / 'samples/labels.csv')

    def test_dataset_and_both_splits(self):
        result = verify(ROOT)
        self.assertEqual(result['image_count'], 202)
        self.assertEqual(result['unique_image_hashes'], 202)
        for file, counts in [('Datasets/fixed_split_seed21.json', [145, 26, 31]),
                             ('Experiments/profile_split_C.json', [143, 20, 39])]:
            config = RunParameters(split=str(ROOT / file))
            loaders, groups = prepare_loaders(config)
            self.assertEqual([len(groups[k]) for k in ('train', 'val', 'test')], counts)
            self.assertEqual(len(loaders['test'].dataset), counts[2])

    def test_overlap_is_rejected(self):
        split = json.loads((ROOT / 'Datasets/fixed_split_seed21.json').read_text())
        split['val'][0] = split['train'][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'split.json'
            path.write_text(json.dumps(split))
            with self.assertRaises(ValueError):
                read_split(path)

    def test_augmentation_preserves_texture_and_targets(self):
        batch1 = read_image_batch(self.records[:2], training=False)
        batch2 = read_image_batch(self.records[:2], training=True)
        self.assertEqual(tuple(batch1[0].shape), (2, 3, 224, 224))
        self.assertEqual(tuple(batch1[1].shape), (2, 944))
        torch.testing.assert_close(batch1[1], batch2[1], rtol=0, atol=0)
        torch.testing.assert_close(batch1[2], batch2[2], rtol=0, atol=0)
        torch.testing.assert_close(batch1[2].sum(1), torch.ones(2))

    def test_padding_is_excluded_from_histograms(self):
        image = Image.fromarray(np.arange(120 * 60 * 3, dtype=np.uint8).reshape(120, 60, 3))
        prepared = ResizePad224()(image)
        changed = prepared.padded_gray.copy()
        changed[~prepared.content_mask] = 255
        extractor = UniformHMSLBPFeatureExtractor()
        first, _ = extractor.extract_prepared(prepared.padded_rgb, prepared.content_mask, prepared.padded_gray)
        second, _ = extractor.extract_prepared(prepared.padded_rgb, prepared.content_mask, changed)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(UNIFORM_U2_LOOKUP)), 59)

    def test_metric_formula(self):
        target = np.array([[0, .1, .2, .3, .4], [.1, .2, .3, .1, .3]])
        prediction = target + np.array([[.01, -.01, 0, 0, 0], [0, 0, -.02, .02, 0]])
        result = distribution_metrics(target, prediction)
        expected_chi = np.mean(.5 * np.sum((prediction-target)**2 / (prediction+target+1e-8), axis=1))
        expected_nrmse = 100 * np.mean(np.sqrt(np.mean((prediction-target)**2, axis=0)) / (np.ptp(target, axis=0)+1e-8))
        self.assertAlmostEqual(result['chi_square'], expected_chi)
        self.assertAlmostEqual(result['mae'], .006)
        self.assertAlmostEqual(result['nrmse'], expected_nrmse)

    def test_backward_and_checkpoint_roundtrip(self):
        image, hms, target, _ = read_image_batch(self.records[:2])
        model = MTA_Net(pretrained=False)
        loss = PaperDistributionLoss()(model(image, hms), target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.hms_projection.gamma.grad)
        self.assertTrue(torch.isfinite(model.hms_projection.gamma.grad))
        model.eval()
        with torch.no_grad():
            expected = model(image, hms)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'model.pt'
            split = read_split(ROOT / 'Datasets/fixed_split_seed21.json')
            save_checkpoint(file, model, 1, {}, RunParameters(), split)
            restored, _ = load_checkpoint(file)
            restored.eval()
            with torch.no_grad():
                actual = restored(image, hms)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_batchnorm_sampler(self):
        sampler = SafeBatchSampler(range(145), 8, seed=21)
        first, second = list(sampler), list(sampler)
        self.assertEqual(len(first), len(sampler))
        self.assertEqual(sum(map(len, first)), 144)
        self.assertTrue(all(len(batch) >= 2 for batch in first))
        self.assertNotEqual(first, second)


if __name__ == '__main__':
    unittest.main()
