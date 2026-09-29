"""Tests for sav_e -- EE AVE."""

from morie.fn import _array_core as np
from morie.fn._containers import ESRes
from morie.fn.sav_e import subscale_ee_ave


class TestSavE:
    def test_basic(self, mapq_df):
        result = subscale_ee_ave(mapq_df)
        assert isinstance(result, ESRes)
        assert 0 < result.estimate <= 1

    def test_array_input(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, (100, 5))
        result = subscale_ee_ave(X, items=None)
        assert result.estimate > 0


def _three_item_rows():
    """Three items: the one-factor model is just-identified, so the ML
    loadings are lambda_1^2 = r12 r13 / r23 (and cyclically)."""
    f = [0.3, -1.2, 0.8, 1.5, -0.4, 0.1, -2.0, 1.1, 0.6, -0.9, 0.0, 1.9]
    e1 = [0.5, 0.1, -0.7, 0.2, 0.9, -0.3, 0.4, -0.6, 0.2, 0.8, -1.1, 0.3]
    e2 = [-0.2, 0.6, 0.3, -0.9, 0.1, 0.7, -0.5, 0.4, -0.8, 0.2, 0.9, -0.1]
    e3 = [0.9, -0.4, 0.2, 0.3, -0.6, -0.2, 0.8, 0.1, 0.5, -0.7, 0.3, 0.6]
    return [[f[i] + e1[i], 0.8 * f[i] + e2[i], f[i] + 2.0 * e3[i]] for i in range(12)]


def _corr(rows):
    import math

    n = len(rows)
    m = [sum(r[j] for r in rows) / n for j in range(3)]
    c = [[sum((r[a] - m[a]) * (r[b] - m[b]) for r in rows) for b in range(3)] for a in range(3)]
    return [[c[a][b] / math.sqrt(c[a][a] * c[b][b]) for b in range(3)] for a in range(3)]


def test_one_factor_ml_loadings_and_ave_recomputed():
    import math

    import pytest

    rows = _three_item_rows()
    R = _corr(rows)
    lam = [
        math.sqrt(R[0][1] * R[0][2] / R[1][2]),
        math.sqrt(R[0][1] * R[1][2] / R[0][2]),
        math.sqrt(R[0][2] * R[1][2] / R[0][1]),
    ]
    r = subscale_ee_ave(rows, items=None)
    assert r.extra["loadings"] == pytest.approx(lam, rel=1e-9)
    assert r.estimate == pytest.approx(sum(v * v for v in lam) / 3, rel=1e-9)
    assert r.extra["converged"] is True
