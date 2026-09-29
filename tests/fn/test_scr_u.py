"""Tests for scr_u -- UA composite reliability."""

from morie.fn import _array_core as np
from morie.fn._containers import ESRes
from morie.fn.scr_u import subscale_ua_composite_rel


class TestScrU:
    def test_basic(self, mapq_df):
        result = subscale_ua_composite_rel(mapq_df)
        assert isinstance(result, ESRes)
        assert 0 < result.estimate <= 1

    def test_array_input(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, (100, 5))
        result = subscale_ua_composite_rel(X, items=None)
        assert result.estimate > 0


def test_composite_reliability_from_the_ml_loadings():
    """Three items: ML loadings lambda_1^2 = r12 r13 / r23 (just-identified),
    rho_c = (sum lambda)^2 / ((sum lambda)^2 + sum(1 - lambda^2))."""
    import math

    import pytest

    f = [0.3, -1.2, 0.8, 1.5, -0.4, 0.1, -2.0, 1.1, 0.6, -0.9, 0.0, 1.9]
    e1 = [0.5, 0.1, -0.7, 0.2, 0.9, -0.3, 0.4, -0.6, 0.2, 0.8, -1.1, 0.3]
    e2 = [-0.2, 0.6, 0.3, -0.9, 0.1, 0.7, -0.5, 0.4, -0.8, 0.2, 0.9, -0.1]
    e3 = [0.9, -0.4, 0.2, 0.3, -0.6, -0.2, 0.8, 0.1, 0.5, -0.7, 0.3, 0.6]
    rows = [[f[i] + e1[i], 0.8 * f[i] + e2[i], f[i] + 2.0 * e3[i]] for i in range(12)]
    n = 12
    m = [sum(r[j] for r in rows) / n for j in range(3)]
    c = [[sum((r[a] - m[a]) * (r[b] - m[b]) for r in rows) for b in range(3)] for a in range(3)]
    R = [[c[a][b] / math.sqrt(c[a][a] * c[b][b]) for b in range(3)] for a in range(3)]
    lam = [
        math.sqrt(R[0][1] * R[0][2] / R[1][2]),
        math.sqrt(R[0][1] * R[1][2] / R[0][2]),
        math.sqrt(R[0][2] * R[1][2] / R[0][1]),
    ]
    s = sum(lam)
    want = s * s / (s * s + sum(1 - v * v for v in lam))
    assert subscale_ua_composite_rel(rows).estimate == pytest.approx(want, rel=1e-9)
