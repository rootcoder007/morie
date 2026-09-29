"""Tests for covar (sample covariance)."""

import math

import pytest

from morie.fn.covar import covar


def test_covariance_and_correlation_recomputed():
    x = [1.0, 2.5, 3.0, 4.5, 2.0]
    y = [2.0, 2.0, 5.0, 4.0, 1.5]
    mx, my = sum(x) / 5, sum(y) / 5
    c = sum((a - mx) * (b - my) for a, b in zip(x, y)) / 4
    sx = math.sqrt(sum((a - mx) ** 2 for a in x) / 4)
    sy = math.sqrt(sum((b - my) ** 2 for b in y) / 4)
    r = covar(x, y)
    assert r["value"] == pytest.approx(c, rel=1e-13)
    assert r["correlation"] == pytest.approx(c / (sx * sy), rel=1e-12)
    with pytest.raises(ValueError):
        covar([1.0, 2.0], [1.0])
