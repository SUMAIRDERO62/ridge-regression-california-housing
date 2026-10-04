"""Offline tests (synthetic data, no download needed). Run with: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import add_age_group, inject_missing  # noqa: E402
from src.experiment import run_experiment  # noqa: E402
from src.pipeline import build_pipeline, split_columns  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402


def make_synthetic(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "MedInc": rng.uniform(1, 10, n),
        "HouseAge": rng.uniform(1, 52, n),
        "AveRooms": rng.uniform(3, 8, n),
        "AveBedrms": rng.uniform(0.8, 1.5, n),
        "Population": rng.uniform(300, 3000, n),
        "AveOccup": rng.uniform(1.5, 5, n),
        "Latitude": rng.uniform(32, 42, n),
        "Longitude": rng.uniform(-124, -114, n),
    })
    y = pd.Series(0.4 * X["MedInc"] + 0.01 * X["HouseAge"] + rng.normal(0, 0.3, n))
    return X, y


def test_add_age_group_creates_categories():
    X, _ = make_synthetic()
    out = add_age_group(X)
    assert set(out["AgeGroup"].unique()) <= {"new", "mid", "old", "very_old"}
    assert "AgeGroup" not in X.columns  # original is untouched


def test_inject_missing_fraction_is_close():
    X, _ = make_synthetic(n=5000)
    out = inject_missing(X, fraction=0.05)
    rate = out["AveRooms"].isna().mean()
    assert 0.03 < rate < 0.07


def test_pipeline_handles_missing_and_categorical():
    X, y = make_synthetic()
    X = inject_missing(add_age_group(X), fraction=0.05)
    num, cat = split_columns(X)
    assert cat == ["AgeGroup"]
    pipe = build_pipeline(num, cat, Ridge()).fit(X, y)
    assert np.isfinite(pipe.predict(X)).all()


def test_run_experiment_creates_artifacts(tmp_path):
    X, y = make_synthetic()
    X = inject_missing(add_age_group(X), fraction=0.03)
    res = run_experiment(X, y, alphas=[0.1, 1, 10], cv_folds=3, out_dir=tmp_path)
    for name in ["metrics.json", "ridge_model.joblib", "pred_vs_actual.png",
                 "residuals.png", "coefficients.png", "alpha_curve.png"]:
        assert (tmp_path / name).exists(), name
    ridge = res["models"]["Ridge (tuned)"]["test"]
    dummy = res["models"]["Dummy (mean)"]["test"]
    assert ridge["r2"] > 0.5
    assert ridge["rmse"] < dummy["rmse"]
