"""Prepare RGB, HMS-LBP and measured-composition tensors for a batch."""
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from Parameters import COLOR_JITTER, IMAGENET_MEAN, IMAGENET_STD
from Utils.HMS_LBP import ResizePad224, UniformHMSLBPFeatureExtractor


def read_image_batch(records, training=False):
    """Apply colour augmentation to RGB only; compute HMS from original luminance."""
    resize_pad = ResizePad224()
    extractor = UniformHMSLBPFeatureExtractor()
    jitter = transforms.ColorJitter(
        brightness=COLOR_JITTER, contrast=COLOR_JITTER, saturation=COLOR_JITTER
    )
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    images, textures, targets, ids = [], [], [], []
    for record in records:
        with Image.open(record.image_path) as source:
            prepared = resize_pad(source)
        hms, _ = extractor.extract_prepared(
            prepared.padded_rgb, prepared.content_mask, prepared.padded_gray
        )
        rgb = jitter(prepared.padded_rgb) if training else prepared.padded_rgb
        array = np.array(rgb, dtype=np.float32, copy=True) / 255.0
        images.append(normalize(torch.from_numpy(array).permute(2, 0, 1)))
        textures.append(torch.from_numpy(hms))
        targets.append(torch.tensor(record.target, dtype=torch.float32))
        ids.append(record.sample_id)
    return torch.stack(images), torch.stack(textures), torch.stack(targets), ids
