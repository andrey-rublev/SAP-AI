"""Tests for the plotting helpers."""
import numpy as np
import pytest

from metrics import moving_average, plot_training_curve


def test_moving_average_smooths():
    avg = moving_average([0, 0, 0, 3, 3, 3], window=3)
    assert len(avg) == 4  # len - window + 1
    assert avg[0] == pytest.approx(0.0)
    assert avg[-1] == pytest.approx(3.0)


def test_moving_average_passthrough_when_short():
    values = [1, 2]
    assert np.array_equal(moving_average(values, window=5), np.asarray(values, dtype=float))


def test_plot_training_curve_writes_png(tmp_path):
    path = tmp_path / "curve.png"
    history = list(np.linspace(-5, 10, 300))
    out = plot_training_curve(history, str(path), window=50)
    assert out == str(path)
    assert path.exists()
    assert path.stat().st_size > 0
