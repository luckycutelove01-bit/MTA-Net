
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

def normalize_distribution(preds, eps=1e-8):
    preds = np.asarray(preds, dtype=np.float32)
    preds = np.clip(preds, 0.0, None)
    row_sum = preds.sum(axis=1, keepdims=True)
    zero_rows = row_sum.squeeze(1) <= eps
    if np.any(zero_rows):
        preds[zero_rows] = 1.0 / preds.shape[1]
        row_sum = preds.sum(axis=1, keepdims=True)
    return preds / (row_sum + eps)


def glcm_features(gray, levels=16):
    q = np.clip((gray * levels).astype(np.int32), 0, levels - 1)
    distances = [1, 2, 4]
    offsets = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    feats = []
    i_idx = np.arange(levels, dtype=np.float32)[:, None]
    j_idx = np.arange(levels, dtype=np.float32)[None, :]
    for d in distances:
        for dy, dx in offsets:
            y0 = max(0, -dy * d)
            y1 = q.shape[0] - max(0, dy * d)
            x0 = max(0, -dx * d)
            x1 = q.shape[1] - max(0, dx * d)
            a = q[y0:y1, x0:x1].ravel()
            b = q[y0 + dy * d:y1 + dy * d, x0 + dx * d:x1 + dx * d].ravel()
            mat = np.zeros((levels, levels), dtype=np.float32)
            np.add.at(mat, (a, b), 1.0)
            mat = mat + mat.T
            mat = mat / (mat.sum() + 1e-8)
            diff2 = (i_idx - j_idx) ** 2
            contrast = float(np.sum(diff2 * mat))
            dissimilarity = float(np.sum(np.abs(i_idx - j_idx) * mat))
            homogeneity = float(np.sum(mat / (1.0 + diff2)))
            asm = float(np.sum(mat ** 2))
            energy = float(np.sqrt(asm))
            mu_i = float(np.sum(i_idx * mat))
            mu_j = float(np.sum(j_idx * mat))
            sig_i = float(np.sqrt(np.sum(((i_idx - mu_i) ** 2) * mat)))
            sig_j = float(np.sqrt(np.sum(((j_idx - mu_j) ** 2) * mat)))
            corr = float(np.sum((i_idx - mu_i) * (j_idx - mu_j) * mat) / (sig_i * sig_j + 1e-8))
            feats.extend([contrast, dissimilarity, homogeneity, asm, energy, corr])
    return np.asarray(feats, dtype=np.float32)


def haar_features(gray):
    feats = []
    current = gray.astype(np.float32)
    for _ in range(3):
        h, w = current.shape
        h -= h % 2
        w -= w % 2
        current = current[:h, :w]
        a = current[0::2, 0::2]
        b = current[0::2, 1::2]
        c = current[1::2, 0::2]
        d = current[1::2, 1::2]
        bands = [
            (a + b + c + d) * 0.25,
            (a + b - c - d) * 0.25,
            (a - b + c - d) * 0.25,
            (a - b - c + d) * 0.25,
        ]
        for band in bands:
            feats.extend([
                float(np.mean(band)),
                float(np.std(band)),
                float(np.mean(np.abs(band))),
                float(np.percentile(band, 25)),
                float(np.percentile(band, 75)),
            ])
        current = bands[0]
    return np.asarray(feats, dtype=np.float32)


def build_rf(seed=21):
    # Random-forest estimator settings.
    return MultiOutputRegressor(RandomForestRegressor(
        n_estimators=500, max_features='sqrt', min_samples_leaf=2,
        random_state=int(seed), n_jobs=-1))


def rf_features(gray, kind, content_mask=None):
    """Accept a supplied float grayscale array in [0,1]; no file I/O."""
    if kind == 'glcm_rf':
        return glcm_features(gray)
    if kind == 'wavelet_rf':
        return haar_features(gray)
    if kind == 'lbp_rf':
        from Experiments.texture_encodings import TraditionalLBPFeatureExtractor
        mask = np.ones_like(gray, dtype=bool) if content_mask is None else content_mask
        rgb = np.repeat(gray[..., None], 3, axis=2)
        feature, _ = TraditionalLBPFeatureExtractor().extract_prepared(rgb, mask, gray * 255.0)
        return feature
    raise ValueError(kind)
