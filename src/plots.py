"""Diagnostic plots (saved to disk; safe on headless machines)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_pred_vs_actual(y_true, y_pred, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.35)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set(xlabel="Actual value ($100k)", ylabel="Predicted value ($100k)",
           title="Predicted vs Actual (test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, path: Path) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_pred, residuals, s=6, alpha=0.35)
    axes[0].axhline(0, color="r", linestyle="--")
    axes[0].set(xlabel="Predicted value", ylabel="Residual", title="Residuals vs Predicted")
    axes[1].hist(residuals, bins=50, edgecolor="black")
    axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coefficients(pipeline, path: Path, top_n: int = 15) -> None:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    coefs = np.asarray(pipeline.named_steps["model"].coef_).ravel()
    order = np.argsort(np.abs(coefs))[::-1][:top_n][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([names[i] for i in order], coefs[order],
            color=["tab:green" if coefs[i] > 0 else "tab:red" for i in order])
    ax.set(xlabel="Coefficient (standardised features)", title="Ridge coefficients (largest magnitude)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_alpha_curve(cv_results: dict, path: Path) -> None:
    alphas = np.array(cv_results["param_model__alpha"], dtype=float)
    rmse = -np.array(cv_results["mean_test_score"])
    std = np.array(cv_results["std_test_score"])
    order = np.argsort(alphas)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(alphas[order], rmse[order], yerr=std[order], marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set(xlabel="alpha (log scale)", ylabel="CV RMSE", title="Ridge: regularisation strength vs CV RMSE")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
