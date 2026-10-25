
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from Parameters import IMAGE_SIZE,PADDING_RGB,HMS_RADII,HMS_GRID_SIZE,HMS_NUM_BINS,HMS_FEATURE_DIM
def hms_offsets(radius: int) -> list[tuple[int, int]]:
    """Manuscript equations (6)-(7): b0..b7 order, as (dx, dy)."""
    r = int(radius)
    return [
        (-2 * r, 0), (-r, 0), (r, 0), (2 * r, 0),
        (-2 * r, -r), (2 * r, -r), (-2 * r, r), (2 * r, r),
    ]

def transition_count(raw_code: int) -> int:
    bits = [(int(raw_code) >> bit) & 1 for bit in range(8)]
    return sum(abs(bits[bit] - bits[(bit + 1) % 8]) for bit in range(8))

def build_uniform_u2_lookup() -> tuple[np.ndarray, list[int]]:
    uniform_codes = [code for code in range(256) if transition_count(code) <= 2]
    assert len(uniform_codes) == 58
    lookup = np.full(256, 58, dtype=np.int16)
    for bin_id, raw_code in enumerate(uniform_codes):
        lookup[raw_code] = bin_id
    assert lookup.shape == (256,)
    assert int(lookup.min()) == 0
    assert int(lookup.max()) == 58
    return lookup, uniform_codes

UNIFORM_U2_LOOKUP, UNIFORM_U2_CODES = build_uniform_u2_lookup()

@dataclass(frozen=True)
class ResizePadResult:
    padded_rgb: Image.Image
    content_mask: np.ndarray
    padded_gray: np.ndarray

class ResizePad224:
    def __init__(self, size: int = IMAGE_SIZE, fill: tuple[int, int, int] = PADDING_RGB):
        self.size = int(size)
        self.fill = tuple(int(v) for v in fill)

    def __call__(self, image: Image.Image) -> ResizePadResult:
        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {image.size}")
        scale = min(self.size / width, self.size / height)
        new_width = max(1, min(self.size, int(round(width * scale))))
        new_height = max(1, min(self.size, int(round(height * scale))))
        resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2
        canvas.paste(resized, (left, top))
        content_mask = np.zeros((self.size, self.size), dtype=bool)
        content_mask[top:top + new_height, left:left + new_width] = True
        # Texture branch: original RGB -> float luminance -> bilinear resize.
        # Retain the established coefficients/interpolation, without RGB uint8
        # resize rounding occurring before grayscale conversion.
        gray = Image.fromarray(rgb_to_luminance(image))
        resized_gray = np.asarray(gray.resize((new_width, new_height), Image.Resampling.BILINEAR), dtype=np.float32)
        gray_fill = float(np.dot(np.asarray(self.fill), [0.299, 0.587, 0.114]))
        padded_gray = np.full((self.size, self.size), gray_fill, dtype=np.float32)
        padded_gray[top:top + new_height, left:left + new_width] = resized_gray
        return ResizePadResult(canvas, content_mask, padded_gray)

def rgb_to_luminance(padded_rgb: Image.Image | np.ndarray) -> np.ndarray:
    rgb = np.asarray(padded_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB array [H,W,3], got {rgb.shape}")
    return (
        np.float32(0.299) * rgb[..., 0]
        + np.float32(0.587) * rgb[..., 1]
        + np.float32(0.114) * rgb[..., 2]
    ).astype(np.float32, copy=False)

def encode_raw_code_at(gray: np.ndarray, x: int, y: int, radius: int) -> int:
    center = gray[y, x]
    raw_code = 0
    for bit, (dx, dy) in enumerate(hms_offsets(radius)):
        if gray[y + dy, x + dx] >= center:
            raw_code |= 1 << bit
    return raw_code

def encode_hms_scale(gray: np.ndarray, content_mask: np.ndarray, radius: int,
                     lookup: np.ndarray = UNIFORM_U2_LOOKUP) -> tuple[np.ndarray, np.ndarray]:
    """Encode one HMS scale; invalid/padding centers remain -1 and never enter bins."""
    gray = np.asarray(gray, dtype=np.float32)
    content_mask = np.asarray(content_mask, dtype=bool)
    if gray.shape != content_mask.shape:
        raise ValueError(f"gray/content_mask shape mismatch: {gray.shape} vs {content_mask.shape}")
    height, width = gray.shape
    yy, xx = np.indices((height, width))
    valid = content_mask.copy()
    raw_codes = np.zeros((height, width), dtype=np.uint16)
    for bit, (dx, dy) in enumerate(hms_offsets(radius)):
        neighbor_x = xx + dx
        neighbor_y = yy + dy
        inside = (
            (neighbor_x >= 0) & (neighbor_x < width)
            & (neighbor_y >= 0) & (neighbor_y < height)
        )
        clipped_x = np.clip(neighbor_x, 0, width - 1)
        clipped_y = np.clip(neighbor_y, 0, height - 1)
        neighbor_in_content = content_mask[clipped_y, clipped_x] & inside
        bit_is_one = (gray[clipped_y, clipped_x] >= gray) & inside
        valid &= neighbor_in_content
        raw_codes |= (bit_is_one.astype(np.uint16) << bit)
    code_map = np.full((height, width), -1, dtype=np.int16)
    code_map[valid] = lookup[raw_codes[valid]]
    return code_map, valid

def local_frequency_histogram(code_map: np.ndarray, valid_mask: np.ndarray,
                              grid_size: int = HMS_GRID_SIZE,
                              num_bins: int = HMS_NUM_BINS) -> tuple[np.ndarray, list[str]]:
    code_map = np.asarray(code_map)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if code_map.shape != valid_mask.shape:
        raise ValueError("code_map and valid_mask must have the same shape")
    height, width = code_map.shape
    grid_h, grid_w = height // grid_size, width // grid_size
    histograms: list[np.ndarray] = []
    warnings: list[str] = []
    for grid_y in range(grid_size):
        for grid_x in range(grid_size):
            y1 = grid_y * grid_h
            y2 = (grid_y + 1) * grid_h if grid_y < grid_size - 1 else height
            x1 = grid_x * grid_w
            x2 = (grid_x + 1) * grid_w if grid_x < grid_size - 1 else width
            block_valid = valid_mask[y1:y2, x1:x2]
            block_codes = code_map[y1:y2, x1:x2][block_valid]
            if block_codes.size == 0:
                histogram = np.zeros(num_bins, dtype=np.float32)
                warnings.append(f"empty_valid_region:grid_y={grid_y},grid_x={grid_x}")
            else:
                counts = np.bincount(block_codes.astype(np.int64), minlength=num_bins)
                histogram = (counts / float(block_codes.size)).astype(np.float32)
                if not np.isclose(histogram.sum(), 1.0, atol=1e-5):
                    raise RuntimeError("Local HMS histogram does not sum to one")
            histograms.append(histogram)
    return np.concatenate(histograms).astype(np.float32, copy=False), warnings

class UniformHMSLBPFeatureExtractor(nn.Module):
    """Non-trainable, deterministic paper-aligned HMS-LBP extractor."""

    def __init__(self, grid_size: int = HMS_GRID_SIZE,
                 radii: Sequence[int] = HMS_RADII,
                 num_bins: int = HMS_NUM_BINS):
        super().__init__()
        if int(num_bins) != 59:
            raise ValueError("Uniform-u2 HMS-LBP requires 59 bins")
        self.grid_size = int(grid_size)
        self.radii = tuple(int(r) for r in radii)
        self.num_bins = int(num_bins)
        self.feature_dim = self.num_bins * self.grid_size * self.grid_size

    def extract_prepared(self, padded_rgb: Image.Image | np.ndarray,
                         content_mask: np.ndarray,
                         padded_gray: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
        # Two-argument use remains a low-level already-prepared-image API.
        # Formal image ingestion must use extract_pil or provide padded_gray.
        gray = rgb_to_luminance(padded_rgb) if padded_gray is None else np.asarray(padded_gray, dtype=np.float32)
        scale_features = []
        all_warnings: list[str] = []
        for radius in self.radii:
            code_map, valid = encode_hms_scale(gray, content_mask, radius)
            feature, warnings = local_frequency_histogram(
                code_map, valid, self.grid_size, self.num_bins
            )
            scale_features.append(feature)
            all_warnings.extend(f"r={radius}:{warning}" for warning in warnings)
        hms = np.mean(np.stack(scale_features, axis=0), axis=0).astype(np.float32)
        if hms.shape != (self.feature_dim,):
            raise RuntimeError(f"Unexpected HMS feature shape: {hms.shape}")
        return hms, all_warnings

    def extract_pil(self, image: Image.Image,
                    resize_pad: ResizePad224 | None = None) -> tuple[np.ndarray, list[str]]:
        prepared = (resize_pad or ResizePad224())(image)
        return self.extract_prepared(prepared.padded_rgb, prepared.content_mask, prepared.padded_gray)

    def forward(self, rgb_batch: torch.Tensor, content_masks: torch.Tensor) -> torch.Tensor:
        if rgb_batch.dim() != 4 or rgb_batch.shape[1] != 3:
            raise ValueError(f"Expected unnormalized RGB [B,3,H,W], got {rgb_batch.shape}")
        if content_masks.shape != (rgb_batch.shape[0], rgb_batch.shape[2], rgb_batch.shape[3]):
            raise ValueError("content_masks must have shape [B,H,W]")
        features = []
        for index in range(rgb_batch.shape[0]):
            rgb = rgb_batch[index].detach().cpu().permute(1, 2, 0).numpy()
            mask = content_masks[index].detach().cpu().numpy().astype(bool)
            feature, _ = self.extract_prepared(rgb, mask)
            features.append(torch.from_numpy(feature))
        return torch.stack(features, dim=0).to(rgb_batch.device)

