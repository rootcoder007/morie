"""Tests for johw."""

import numpy as np
import pytest

from morie.fn.johw import joseph_holt_winters

def test_johw_basic():
    m = 12
    season = np.array([3., 1., -2., -4., -1., 2., 5., 4., 1., -1., -3., -5.])
    rng = np.random.default_rng(0)
    y = np.concatenate(
        [10 + 0.5 * np.arange(i * m, (i + 1) * m) + season for i in range(8)]
    ) + rng.normal(0, 0.5, 96)
    fc = joseph_holt_winters(y, m=m, horizon=12)["forecast"]
    t = np.arange(12.0)
    fc = fc - np.polyval(np.polyfit(t, fc, 1), t)  # remove the trend it carries
    assert np.corrcoef(fc, season)[0, 1] > 0.9  # measured 0.954


def test_johw_edge():
    y = np.abs(np.sin(np.arange(48) / 3.0)) + 2.0
    with pytest.raises(ValueError):
        joseph_holt_winters(y[:10], m=12)  # fewer than 2 periods
    with pytest.raises(ValueError):
        # multiplicative refuses non-positive data rather than returning inf
        joseph_holt_winters(np.r_[y, 0.0], m=12, seasonal="multiplicative")
