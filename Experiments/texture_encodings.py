"""LBP variants sharing grayscale preprocessing, valid masks and histogram layout."""
import numpy as np

from Parameters import HMS_GRID_SIZE, HMS_NUM_BINS
from Utils.HMS_LBP import (
    UniformHMSLBPFeatureExtractor, UNIFORM_U2_LOOKUP, encode_lbp_scale,
)


def circular_offsets(radius):
    r = int(radius)
    return [(-r, -r), (0, -r), (r, -r), (r, 0),
            (r, r), (0, r), (-r, r), (-r, 0)]


def vertical_offsets(radius):
    r = int(radius)
    return [(0, -2*r), (0, -r), (0, r), (0, 2*r),
            (-r, -2*r), (r, -2*r), (-r, 2*r), (r, 2*r)]


def rotation_lookup():
    """Canonicalize circular bit rotations before the uniform-u2 lookup."""
    canonical = [min(((code >> shift) | (code << (8-shift))) & 255
                     for shift in range(8)) for code in range(256)]
    return UNIFORM_U2_LOOKUP[np.asarray(canonical)]


class LBPFeatureExtractor(UniformHMSLBPFeatureExtractor):
    """Horizontal multiscale LBP used by MTA-Net."""


class TraditionalLBPFeatureExtractor(UniformHMSLBPFeatureExtractor):
    encoding = 'standard'
    offsets = staticmethod(circular_offsets)
    lookup = UNIFORM_U2_LOOKUP

    def __init__(self, grid_size=HMS_GRID_SIZE, radius=1, num_bins=HMS_NUM_BINS):
        super().__init__(grid_size=grid_size, radii=(radius,), num_bins=num_bins)

    def _encode_scale(self, gray, mask, radius):
        return encode_lbp_scale(gray, mask, self.offsets(radius), self.lookup)


class RotationInvariantLBPFeatureExtractor(TraditionalLBPFeatureExtractor):
    encoding = 'rotation_invariant'
    lookup = rotation_lookup()


class VerticalLBPFeatureExtractor(UniformHMSLBPFeatureExtractor):
    encoding = 'vertical'

    def _encode_scale(self, gray, mask, radius):
        return encode_lbp_scale(gray, mask, vertical_offsets(radius))


class IsotropicMultiScaleLBPFeatureExtractor(UniformHMSLBPFeatureExtractor):
    encoding = 'isotropic'

    def _encode_scale(self, gray, mask, radius):
        return encode_lbp_scale(gray, mask, circular_offsets(radius))
