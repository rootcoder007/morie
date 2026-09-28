"""assocstats: association measures recomputed from their definitions."""

import math

import pytest

from morie.fn._rrng_core import pnorm, qnorm
from morie.fn.assocstats import affine_moments, tost_correlation, variation_ratio, yule_association


def test_variation_ratio():
    x = ["theft", "assault", "theft", "fraud", "theft", "assault", "fraud", "fraud", "fraud"]
    assert variation_ratio(x) == pytest.approx(1 - 4 / 9, abs=1e-15)
    assert variation_ratio([7, 7, 7]) == 0.0
    with pytest.raises(ValueError):
        variation_ratio([])


def test_yule_q_and_y_are_linked():
    a, b, c, d = 23.0, 7.0, 11.0, 19.0
    r = yule_association([[a, b], [c, d]])
    assert pytest.approx((a * d - b * c) / (a * d + b * c), rel=1e-14) == r.Q
    assert pytest.approx(2 * r.Y / (1 + r.Y**2), rel=1e-13) == r.Q
    s = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    assert r.se_Q == pytest.approx((1 - r.Q**2) * s / 2, rel=1e-14)
    z = qnorm(0.975)
    assert r.ci_Y[1] - r.ci_Y[0] == pytest.approx(2 * z * (1 - r.Y**2) * s / 4, rel=1e-12)


def test_tost_correlation():
    r, n, lo, hi = 0.08, 120, -0.25, 0.3
    t = tost_correlation(r, n, lo, hi)
    fz = 0.5 * math.log((1 + r) / (1 - r))
    zl = (fz - 0.5 * math.log((1 + lo) / (1 - lo))) * math.sqrt(n - 3)
    zh = (fz - 0.5 * math.log((1 + hi) / (1 - hi))) * math.sqrt(n - 3)
    assert t.p_value == pytest.approx(max(1 - pnorm(zl), pnorm(zh)), rel=1e-10)
    assert t.equivalent == (t.p_value < 0.05)
    # the 90 percent interval lies inside the bounds exactly when both one-sided tests reject at 5 percent
    assert (lo < t.ci[0] and t.ci[1] < hi) == t.equivalent
    with pytest.raises(ValueError):
        tost_correlation(0.1, 3, -0.2, 0.2)


def test_affine_moments():
    mu = [1.0, -2.0, 0.5]
    S = [[2.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.5]]
    A = [[1.0, -1.0, 0.0], [0.5, 0.5, 2.0]]
    B = [[0.0, 1.0, 1.0]]
    r = affine_moments(mu, S, A, [1.0, 2.0], B, [4.0])
    assert r.mean == pytest.approx([1 + 2 + 1, 0.5 - 1 + 1 + 2], rel=1e-14)
    for i in range(2):
        ref = sum(A[i][j] * S[j][k] * B[0][k] for j in range(3) for k in range(3))
        assert r.cov[i][0] == pytest.approx(ref, rel=1e-13)
    assert r.mean_B == [-2.0 + 0.5 + 4.0]
