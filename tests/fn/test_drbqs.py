"""Tests for drbqs.dr_did_quantile."""

import numpy as np
import pytest

from morie.fn.drbqs import dr_did_quantile


def test_drbqs_basic():
    rng = np.random.default_rng(42)
    n = 2500
    x = rng.normal(size=n)
    D = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(float)
    y_pre = 0.5 * x + rng.normal(size=n)
    y_post = y_pre + 0.6 * x + 2.0 * D + rng.normal(scale=0.3, size=n)
    result = dr_did_quantile(y_pre, y_post, D, x, quantile=0.5)
    assert result["qtt"] == pytest.approx(2.0, abs=0.3)  # measured ~2.0
    assert result["qtt"] == pytest.approx(result["q_treated"] - result["q_control"])


def test_drbqs_edge():
    with pytest.raises(ValueError):
        dr_did_quantile([1.0, 2.0], [1.0, 2.0], [1, 0], [0.0, 1.0], quantile=0.0)
    with pytest.raises(ValueError):
        dr_did_quantile([1.0, 2.0], [1.0, 2.0], [0, 0], [0.0, 1.0])  # no treated
