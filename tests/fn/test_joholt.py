"""Tests for joholt."""

import numpy as np
import pytest

from morie.fn.joholt import joseph_holt_linear

def test_joholt_basic():
    y = np.arange(40, dtype=float) * 2.0 + 5.0
    out = joseph_holt_linear(y, horizon=5)
    assert out["forecast"] == pytest.approx(y[-1] + np.arange(1, 6) * 2.0, rel=0.02)
    assert out["sse"] < 1e-3


def test_joholt_edge():
    y = np.arange(40, dtype=float) * 2.0 + 5.0
    # damping must flatten the long horizon
    assert (
        joseph_holt_linear(y, horizon=30, damped=True, phi=0.9)["forecast"][-1]
        < joseph_holt_linear(y, horizon=30)["forecast"][-1]
    )
    with pytest.raises(ValueError):
        joseph_holt_linear(y, alpha=1.5)
