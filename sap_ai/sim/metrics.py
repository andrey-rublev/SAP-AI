"""Plotting helpers for training runs.

Kept separate so the training code stays free of a hard matplotlib dependency -
:func:`plot_training_curve` imports it lazily and uses the non-interactive Agg
backend so it works headless (CI, servers, this repo's tests).
"""
from __future__ import annotations

import numpy as np


def moving_average(values, window: int):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curve(history, path: str, window: int = 50, title: str = "Training reward"):
    """Save a PNG of per-episode reward plus a moving average. Returns ``path``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting; pip install matplotlib") from exc

    history = np.asarray(history, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history, alpha=0.3, color="#6b7280", label="episode reward")

    if len(history) >= window > 1:
        smoothed = moving_average(history, window)
        xs = np.arange(window - 1, len(history))
        ax.plot(xs, smoothed, color="#2563eb", linewidth=2, label=f"{window}-episode moving avg")

    ax.set_xlabel("episode")
    ax.set_ylabel("total reward")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
