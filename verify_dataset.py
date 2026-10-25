"""Verify image-label correspondence, image hashes and fixed data splits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def verify(root):
    root = Path(root)
    sample_dir = root / 'samples'
    with (sample_dir / 'labels.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    with (sample_dir / 'sample_manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        manifest = list(csv.DictReader(handle))
    ids = [row['Sample_ID'] for row in rows]
    expected = {f'wh1-{i:03d}.tif' for i in range(1, 203)}
    if len(ids) != 202 or set(ids) != expected:
        raise ValueError('Labels must contain exactly one row for each of the 202 image IDs')
    images = {p.name: p for p in (sample_dir / 'images').glob('*.tif')}
    if set(images) != expected:
        raise ValueError('Image IDs do not match the label IDs')
    if len(manifest) != 202 or {r['Sample_ID'] for r in manifest} != expected:
        raise ValueError('Sample manifest IDs do not match the label IDs')
    hashes = {}
    for sid, path in images.items():
        with Image.open(path) as image:
            image.load()
        hashes[sid] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(set(hashes.values())) != 202:
        raise ValueError('Image contents must be distinct')
    if any(hashes[r['Sample_ID']] != r['Image_SHA256'] for r in manifest):
        raise ValueError('Image SHA256 differs from sample_manifest.csv')
    columns = ('lt100', '100_250', '250_500', '500_1000', '1000_2000')
    values = np.array([[float(r[c]) for c in columns] for r in rows])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Labels must be finite and non-negative')
    totals = values.sum(axis=1)
    if not np.allclose(totals, 100, atol=0.01, rtol=0):
        raise ValueError('Measured percentages must sum to 100 within 0.01')
    split_counts = {}
    for name, file, counts in (
        ('main', 'Datasets/fixed_split_seed21.json', [145, 26, 31]),
        ('profile_segments', 'Experiments/profile_split_C.json', [143, 20, 39]),
    ):
        split = json.loads((root / file).read_text(encoding='utf-8-sig'))
        actual = [len(split[k]) for k in ('train', 'val', 'test')]
        flat = sum((split[k] for k in ('train', 'val', 'test')), [])
        if actual != counts or len(flat) != 202 or set(flat) != expected:
            raise ValueError(f'Invalid membership in {file}')
        if name == 'main':
            subset = {sid: key for key in ('train', 'val', 'test') for sid in split[key]}
            if any(subset[r['Sample_ID']] != r['Original_Subset'] for r in manifest):
                raise ValueError('Main split differs from the sample manifest')
        split_counts[name] = dict(zip(('train', 'val', 'test'), actual))
    return {
        'image_count': len(images), 'label_count': len(rows),
        'unique_image_hashes': len(set(hashes.values())),
        'all_images_decoded': True, 'all_image_hashes_match_manifest': True,
        'row_sum_min_pct': float(totals.min()), 'row_sum_max_pct': float(totals.max()),
        'labels_sha256': hashlib.sha256((sample_dir / 'labels.csv').read_bytes()).hexdigest(),
        'split_counts': split_counts,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
