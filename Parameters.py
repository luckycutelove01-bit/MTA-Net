"""Defaults retained from the revised MTA-Net source, not an experiment matrix."""
from dataclasses import dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 224
PADDING_RGB = (128, 128, 128)
HMS_RADII = (1, 2)
HMS_GRID_SIZE = 4
HMS_NUM_BINS = 59
HMS_FEATURE_DIM = 944
HMS_STAGE = 2
CBAM_STAGE = 4
PROJECTION_REDUCTION = 4  # 128 / 4 = 32 hidden channels
GAMMA_INITIAL = 0.10
CBAM_REDUCTION = 16
CBAM_KERNEL = 7
DROPOUT = (0.3, 0.2)
NUM_CLASSES = 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
COLOR_JITTER = 0.1
LOSS_WEIGHTS = (0.6, 0.3, 0.1)
ENTROPY_WEIGHT = 0.6
EPSILON = 1e-8
BIN_NAMES = ['<100 um','100-250 um','250-500 um','500-1000 um','1000-2000 um']
CSV_COLUMNS = ['lt100','100_250','250_500','500_1000','1000_2000']
CHECKPOINT_FORMAT = 'MTA-Net-revised-mask-grayfirst-20261022'

@dataclass
class RunParameters:
    image_dir: str = str(ROOT / 'data' / 'images')
    labels: str = str(ROOT / 'data' / 'grain_size_labels.xlsx')
    split: str = str(ROOT / 'Datasets' / 'fixed_split_seed21.json')
    output: str = str(ROOT / 'results' / 'seed21')
    seed: int = 21  # ONE run; no five-seed launcher
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 8
    gradient_clip: float = 1.0
    num_workers: int = 0
    device: str = 'auto'

def method_settings():
    return {k: v for k, v in globals().items() if k.isupper() and k != 'ROOT'}
