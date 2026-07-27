"""Tests for volrlmt."""

import numpy as np
import pytest

from morie.fn.volrlmt import vol_realised_log_vol_ar


def test_volrlmt_basic():
    rng = np.random.default_rng(42)
    y = np.zeros(400)
    for t in range(1, 400):
        y[t] = 0.1 + 0.7 * y[t - 1] + rng.normal(scale=0.3)
    out = vol_realised_log_vol_ar(np.exp(y))
    assert out["phi"] == pytest.approx(0.7, abs=0.1)
    assert out["forecast"][0] > 0


def test_volrlmt_edge():
    with pytest.raises(ValueError):
        vol_realised_log_vol_ar(np.array([1.0, -1.0] * 10))
    with pytest.raises(ValueError):
        vol_realised_log_vol_ar(np.ones(5))
