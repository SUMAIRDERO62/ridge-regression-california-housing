"""Data loading and light feature preparation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AGE_BINS, AGE_LABELS, MISSING_COLUMNS, RANDOM_STATE, TARGET_NAME


def add_age_group(X: pd.DataFrame) -> pd.DataFrame:
    """Bin HouseAge into a categorical 'AgeGroup' feature.

    California Housing is purely numeric, so this gives the pipeline a real
    categorical column to one-hot encode.
    """
    X = X.copy()
    X["AgeGroup"] = pd.cut(X["HouseAge"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    return X


def inject_missing(
    X: pd.DataFrame,
    columns: list[str] = MISSING_COLUMNS,
    fraction: float = 0.02,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Randomly blank out `fraction` of the cells in `columns` (demo for SimpleImputer)."""
    X = X.copy()
    if fraction <= 0:
        return X
    rng = np.random.default_rng(seed)
    for col in columns:
        mask = rng.random(len(X)) < fraction
        X.loc[mask, col] = np.nan
    return X


def load_housing(missing_fraction: float = 0.02, seed: int = RANDOM_STATE):
    """Download (first run only) and prepare the California Housing dataset."""
    from sklearn.datasets import fetch_california_housing

    housing = fetch_california_housing(as_frame=True)
    X = housing.frame.drop(columns=[TARGET_NAME])
    y = housing.frame[TARGET_NAME]
    X = add_age_group(X)
    X = inject_missing(X, fraction=missing_fraction, seed=seed)
    return X, y
