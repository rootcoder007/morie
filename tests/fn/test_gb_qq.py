"""Tests for gb_qq (Gibbons shelf)."""

import pytest

from morie.fn import _array_core as np
from morie.fn.gb_qq import gibbons_qq_plot


def test_gb_qq_basic():
    rng = np.random.default_rng(5)
    out = gibbons_qq_plot(3.0 + 2.0 * rng.standard_normal(200))
    assert out["slope"] == pytest.approx(2.0, abs=0.25)
    assert out["intercept"] == pytest.approx(3.0, abs=0.25)


def test_gb_qq_edge():
    with pytest.raises(ValueError):
        gibbons_qq_plot([1.0, 2.0])


def test_qq_pairs_and_fitted_line_recomputed():
    """Hazen positions (i - 0.5)/n through the normal quantile; OLS line."""
    from morie.fn._s03core import qnorm

    x = [3.1, -0.4, 2.2, 5.9, 1.7, 0.3]
    n = 6
    q = [qnorm((i - 0.5) / n) for i in range(1, n + 1)]
    xs = sorted(x)
    mq, mx = sum(q) / n, sum(xs) / n
    b = sum((a - mq) * (c - mx) for a, c in zip(q, xs)) / sum((a - mq) ** 2 for a in q)
    r = gibbons_qq_plot(x)
    assert [float(v) for v in r["theoretical"]] == pytest.approx(q, rel=1e-12)
    assert r["slope"] == pytest.approx(b, rel=1e-10)
    assert r["intercept"] == pytest.approx(mx - b * mq, rel=1e-10)
