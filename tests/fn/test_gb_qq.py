"""Tests for gb_qq (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb_qq import gibbons_qq_plot


def test_gb_qq_basic():
    rng = np.random.default_rng(5)
    out = gibbons_qq_plot(3.0 + 2.0 * rng.standard_normal(200))
    assert out["slope"] == pytest.approx(2.0, abs=0.25)
    assert out["intercept"] == pytest.approx(3.0, abs=0.25)


def test_gb_qq_edge():
    with pytest.raises(ValueError):
        gibbons_qq_plot([1.0, 2.0])
