"""Summarize supplied five-run metrics, not images, labels or pooled predictions."""
import numpy as np

SEEDS = (21, 42, 63, 84, 105)

def summarize_five_runs(rows):
    """Rows contain seed, chi_square, mae and nrmse (percent) for ONE configuration.

    Same aggregate mean and ddof=1 convention as train.py:summarize_runs,
    with explicit completeness validation. No file reading or invented runs.
    """
    rows=list(rows)
    if len(rows)!=5 or sorted(row['seed'] for row in rows)!=list(SEEDS):
        raise ValueError('Require exactly one overall result for each of 21,42,63,84,105')
    result={}
    for metric in ('chi_square','mae','nrmse'):
        values=np.asarray([row[metric] for row in rows],dtype=np.float64)
        if values.shape!=(5,) or not np.isfinite(values).all():
            raise ValueError('Require five finite scalar metrics, not per-sample predictions')
        result[metric+'_mean']=float(values.mean())
        result[metric+'_sd']=float(values.std(ddof=1))
    return result
