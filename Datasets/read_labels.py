"""Read measured grain-size percentages and match them to image files."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from Parameters import CSV_COLUMNS


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image_path: Path
    target: tuple[float, ...]


def read_records(image_dir, labels):
    """Return records in label order, with five normalized volume fractions."""
    image_dir, labels = Path(image_dir), Path(labels)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if labels.suffix.lower() == '.csv':
        frame = pd.read_csv(labels, encoding='utf-8-sig', dtype={'Sample_ID': str})
    elif labels.suffix.lower() in ('.xlsx', '.xls'):
        frame = pd.read_excel(labels, dtype={'Sample_ID': str})
    else:
        raise ValueError('Labels must be a CSV or Excel file')
    required = ['Sample_ID', *CSV_COLUMNS]
    absent = sorted(set(required) - set(frame.columns))
    if absent:
        raise ValueError(f"Required label columns are absent: {absent}")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError('Label rows must be nonempty and contain no missing values')
    ids = frame['Sample_ID'].astype(str).str.strip().tolist()
    if len(ids) != len(set(ids)):
        raise ValueError('Sample_ID values must be unique')
    values = frame[CSV_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Measured percentages must be finite and non-negative')
    totals = values.sum(axis=1, keepdims=True)
    if not np.allclose(totals, 100.0, atol=0.01, rtol=0):
        raise ValueError('The five measured percentages must sum to 100 within 0.01')
    targets = values / totals
    records = []
    for sid, target in zip(ids, targets):
        if Path(sid).name != sid or not sid:
            raise ValueError(f"Sample_ID must be an image basename: {sid!r}")
        path = image_dir / sid
        if not path.is_file():
            raise FileNotFoundError(f"Image for {sid} does not exist: {path}")
        records.append(SampleRecord(sid, path, tuple(float(x) for x in target)))
    return records
