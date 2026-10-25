
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class LBPFeatureExtractor(nn.Module):
   
    def __init__(self, grid_size=4, radii=(1, 2, ), num_bins=59):
        super().__init__()
        self.grid_size = grid_size
        self.radii = radii
        self.num_bins = num_bins
        self.feature_dim = num_bins * grid_size * grid_size

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f"LBPFeatureExtractor expects 4D input, got {x.shape}")

        if x.shape[1] > 1:
            x = x[:, 0:1]

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        feats = []
        for i in range(x.shape[0]):
            img = x[i, 0].detach().cpu().numpy()
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            scale_feats = [self._horizontal_lbp(img, r) for r in self.radii]
            feat = torch.stack(scale_feats, dim=0).mean(dim=0)
            feats.append(feat)
        return torch.stack(feats, dim=0).to(x.device)

    def _horizontal_lbp(self, img, radius):
        h, w = img.shape
        r = radius
        lbp_map = np.zeros_like(img, dtype=np.uint8)
        offsets = [
            (-2 * r, 0), (-r, 0), (r, 0), (2 * r, 0),
            (-2 * r, -r), (2 * r, -r), (-2 * r, r), (2 * r, r)
        ]

        for yy in range(2 * r, h - 2 * r):
            for xx in range(2 * r, w - 2 * r):
                center = img[yy, xx]
                code = 0
                for i, (dx, dy) in enumerate(offsets):
                    if img[yy + dy, xx + dx] >= center:
                        code |= (1 << i)
                lbp_map[yy, xx] = code % self.num_bins

        return self._spatial_hist(lbp_map)

    def _spatial_hist(self, lbp):
        h, w = lbp.shape
        gh, gw = h // self.grid_size, w // self.grid_size
        feat = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1 = i * gh
                y2 = (i + 1) * gh if i < self.grid_size - 1 else h
                x1 = j * gw
                x2 = (j + 1) * gw if j < self.grid_size - 1 else w
                block = lbp[y1:y2, x1:x2]
                hist, _ = np.histogram(block, bins=self.num_bins, range=(0, self.num_bins))
                hist = hist.astype(np.float32) / (block.size + 1e-6)
                feat.append(torch.from_numpy(hist))
        return torch.cat(feat, dim=0)


class TraditionalLBPFeatureExtractor(LBPFeatureExtractor):
    def __init__(self, grid_size=4, radius=1, num_bins=59):
        super().__init__(grid_size=grid_size, radii=(radius,), num_bins=num_bins)
        self.radius = radius

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f"TraditionalLBPFeatureExtractor expects 4D input, got {x.shape}")

        if x.shape[1] > 1:
            x = x[:, 0:1]

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        feats = []
        for i in range(x.shape[0]):
            img = x[i, 0].detach().cpu().numpy()
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            feats.append(self._traditional_lbp(img))
        return torch.stack(feats, dim=0).to(x.device)

    def _traditional_lbp(self, img):
        h, w = img.shape
        r = self.radius
        lbp_map = np.zeros_like(img, dtype=np.uint8)
        offsets = [
            (-r, -r), (0, -r), (r, -r), (r, 0),
            (r, r), (0, r), (-r, r), (-r, 0)
        ]

        for yy in range(r, h - r):
            for xx in range(r, w - r):
                center = img[yy, xx]
                code = 0
                for bit, (dx, dy) in enumerate(offsets):
                    if img[yy + dy, xx + dx] >= center:
                        code |= (1 << bit)
                lbp_map[yy, xx] = code % self.num_bins

        return self._spatial_hist(lbp_map)


class RotationInvariantLBPFeatureExtractor(TraditionalLBPFeatureExtractor):
    def _traditional_lbp(self, img):
        h, w = img.shape
        r = self.radius
        lbp_map = np.zeros_like(img, dtype=np.uint8)
        offsets = [
            (-r, -r), (0, -r), (r, -r), (r, 0),
            (r, r), (0, r), (-r, r), (-r, 0)
        ]

        for yy in range(r, h - r):
            for xx in range(r, w - r):
                center = img[yy, xx]
                code = 0
                for bit, (dx, dy) in enumerate(offsets):
                    if img[yy + dy, xx + dx] >= center:
                        code |= (1 << bit)
                rotations = [((code >> shift) | ((code << (8 - shift)) & 0xFF)) for shift in range(8)]
                lbp_map[yy, xx] = min(rotations) % self.num_bins

        return self._spatial_hist(lbp_map)


class VerticalLBPFeatureExtractor(LBPFeatureExtractor):
    def __init__(self, grid_size=4, radii=(1, 2), num_bins=59):
        super().__init__(grid_size=grid_size, radii=radii, num_bins=num_bins)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f"VerticalLBPFeatureExtractor expects 4D input, got {x.shape}")

        if x.shape[1] > 1:
            x = x[:, 0:1]

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        feats = []
        for i in range(x.shape[0]):
            img = x[i, 0].detach().cpu().numpy()
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            scale_feats = [self._vertical_lbp(img, r) for r in self.radii]
            feat = torch.stack(scale_feats, dim=0).mean(dim=0)
            feats.append(feat)
        return torch.stack(feats, dim=0).to(x.device)

    def _vertical_lbp(self, img, radius):
        h, w = img.shape
        r = radius
        lbp_map = np.zeros_like(img, dtype=np.uint8)
        offsets = [
            (0, -2 * r), (0, -r), (0, r), (0, 2 * r),
            (-r, -2 * r), (r, -2 * r), (-r, 2 * r), (r, 2 * r)
        ]

        for yy in range(2 * r, h - 2 * r):
            for xx in range(r, w - r):
                center = img[yy, xx]
                code = 0
                for bit, (dx, dy) in enumerate(offsets):
                    if img[yy + dy, xx + dx] >= center:
                        code |= (1 << bit)
                lbp_map[yy, xx] = code % self.num_bins

        return self._spatial_hist(lbp_map)


class IsotropicMultiScaleLBPFeatureExtractor(TraditionalLBPFeatureExtractor):
    def __init__(self, grid_size=4, radii=(1, 2), num_bins=59):
        super().__init__(grid_size=grid_size, radius=1, num_bins=num_bins)
        self.radii = radii

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f"IsotropicMultiScaleLBPFeatureExtractor expects 4D input, got {x.shape}")

        if x.shape[1] > 1:
            x = x[:, 0:1]

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        feats = []
        for i in range(x.shape[0]):
            img = x[i, 0].detach().cpu().numpy()
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            scale_feats = [self._isotropic_lbp(img, r) for r in self.radii]
            feat = torch.stack(scale_feats, dim=0).mean(dim=0)
            feats.append(feat)
        return torch.stack(feats, dim=0).to(x.device)

    def _isotropic_lbp(self, img, radius):
        old_radius = self.radius
        self.radius = radius
        try:
            return self._traditional_lbp(img)
        finally:
            self.radius = old_radius

