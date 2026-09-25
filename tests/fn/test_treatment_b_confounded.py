"""Tests for treatment_b_confounded (Weisburd et al. 2022, eq. 9.1)."""

import math
import statistics

import pytest

from morie.fn.treatment_b_confounded import treatment_b_confounded


def test_ca9e1_basic():
    """The formula is the OLS coefficient on T in y ~ T + X, written in
    correlations: recompute both on a data set and compare."""
    n = 40
    x = [math.sin(1.1 * k) for k in range(n)]
    t = [0.6 * a + math.cos(2.3 * k) for k, a in enumerate(x)]
    y = [1.0 + 2.0 * b - 0.8 * a + 0.3 * math.sin(5.9 * k) for k, (a, b) in enumerate(zip(x, t))]
    r = statistics.correlation
    s = statistics.stdev
    out = treatment_b_confounded(r(y, t), r(y, x), r(t, x), s(y), s(t))
    # OLS by the normal equations on centred data
    mx, mt, my = map(statistics.fmean, (x, t, y))
    Stt = sum((b - mt) ** 2 for b in t)
    Sxx = sum((a - mx) ** 2 for a in x)
    Stx = sum((a - mx) * (b - mt) for a, b in zip(x, t))
    Sty = sum((b - mt) * (c - my) for b, c in zip(t, y))
    Sxy = sum((a - mx) * (c - my) for a, c in zip(x, y))
    bt = (Sxx * Sty - Stx * Sxy) / (Stt * Sxx - Stx * Stx)
    assert out["value"] == pytest.approx(bt, rel=1e-12)


def test_ca9e1_edge():
    """Uncorrelated confounder: b_t = r_yt s_y/s_t."""
    assert treatment_b_confounded(0.5, 0.3, 0.0, 4.0, 2.0)["value"] == pytest.approx(1.0, abs=1e-15)
