"""End-to-end experiment: baselines, tuned Ridge, metrics, plots, saved model."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from . import plots
from .config import DEFAULT_ALPHAS, RANDOM_STATE
from .evaluate import regression_metrics
from .pipeline import build_pipeline, split_columns

log = logging.getLogger("task2")


def _log_metrics(name: str, split: str, m: dict) -> None:
    log.info("%-18s %-5s | R^2 = %.4f | RMSE = %.4f | MAE = %.4f",
             name, split, m["r2"], m["rmse"], m["mae"])


def run_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    alphas=DEFAULT_ALPHAS,
    test_size: float = 0.2,
    cv_folds: int = 5,
    seed: int = RANDOM_STATE,
    out_dir: str | Path = "results",
    make_plots: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols, categorical_cols = split_columns(X)
    log.info("Rows: %d | numeric: %s | categorical: %s", len(X), numeric_cols, categorical_cols)
    log.info("Missing cells before imputation: %d", int(X.isna().sum().sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    results: dict = {
        "config": {
            "test_size": test_size, "cv_folds": cv_folds, "seed": seed,
            "alphas": list(alphas), "n_train": len(X_train), "n_test": len(X_test),
            "sklearn_version": sklearn.__version__,
        },
        "models": {},
    }

    # ---- Baselines (to prove the model actually learns something) -----------
    for name, estimator in [
        ("Dummy (mean)", DummyRegressor(strategy="mean")),
        ("LinearRegression", LinearRegression()),
    ]:
        pipe = build_pipeline(numeric_cols, categorical_cols, estimator).fit(X_train, y_train)
        m_train = regression_metrics(y_train, pipe.predict(X_train))
        m_test = regression_metrics(y_test, pipe.predict(X_test))
        _log_metrics(name, "train", m_train)
        _log_metrics(name, "test", m_test)
        results["models"][name] = {"train": m_train, "test": m_test}

    # ---- Regularised model: Ridge with CV-tuned alpha -----------------------
    search = GridSearchCV(
        build_pipeline(numeric_cols, categorical_cols, Ridge()),
        param_grid={"model__alpha": list(alphas)},
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        n_jobs=-1,
    ).fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    m_train = regression_metrics(y_train, best.predict(X_train))
    m_test = regression_metrics(y_test, y_pred)
    log.info("Best alpha = %s | CV RMSE = %.4f", search.best_params_["model__alpha"], -search.best_score_)
    _log_metrics("Ridge (tuned)", "train", m_train)
    _log_metrics("Ridge (tuned)", "test", m_test)
    results["models"]["Ridge (tuned)"] = {
        "train": m_train,
        "test": m_test,
        "best_alpha": float(search.best_params_["model__alpha"]),
        "cv_rmse": float(-search.best_score_),
    }

    # ---- Artifacts ----------------------------------------------------------
    joblib.dump(best, out_dir / "ridge_model.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    if make_plots:
        plots.plot_pred_vs_actual(y_test, y_pred, out_dir / "pred_vs_actual.png")
        plots.plot_residuals(y_test, y_pred, out_dir / "residuals.png")
        plots.plot_coefficients(best, out_dir / "coefficients.png")
        plots.plot_alpha_curve(search.cv_results_, out_dir / "alpha_curve.png")

    log.info("Artifacts saved to %s", out_dir.resolve())
    return results
