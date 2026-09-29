"""Tests for em_imputation."""

from morie.fn import _array_core as np
from morie.fn.emimq import em_imputation


class TestEM:
    def test_basic(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (30, 3))
        data[0, 0] = np.nan
        data[5, 2] = np.nan
        r = em_imputation(data)
        assert r.extra["n_missing"] == 2

    def test_means_reasonable(self):
        rng = np.random.default_rng(1)
        data = rng.normal(5, 1, (50, 2))
        data[0, 0] = np.nan
        r = em_imputation(data)
        assert all(3 < m < 7 for m in r.extra["imputed_means"])


def test_monotone_bivariate_pattern_matches_anderson_closed_form():
    """y1 complete, y2 missing for the last rows: the MLE is Anderson's (1957)
    factored-likelihood solution (Little and Rubin 2002, sec. 7.2)."""
    import pytest

    y1 = [1.0, 2.0, 3.5, 4.0, 5.5, 6.0, 2.5, 4.5]
    y2 = [2.1, 2.9, 4.2, 5.1, 6.8, None, None, None]
    n, r = 8, 5
    m1 = sum(y1) / n
    s11 = sum((v - m1) ** 2 for v in y1) / n
    a1 = sum(y1[:r]) / r
    a2 = sum(y2[:r]) / r
    c11 = sum((v - a1) ** 2 for v in y1[:r]) / r
    c12 = sum((y1[i] - a1) * (y2[i] - a2) for i in range(r)) / r
    c22 = sum((v - a2) ** 2 for v in y2[:r]) / r
    b = c12 / c11
    mu2 = a2 + b * (m1 - a1)
    s22 = c22 - b * b * c11 + b * b * s11
    s12 = b * s11
    res = em_imputation([[a, b_] for a, b_ in zip(y1, y2)], max_iter=5000, tol=1e-13)
    assert res.extra["mean"] == pytest.approx([m1, mu2], rel=1e-9)
    cov = res.extra["cov"]
    assert [cov[0][0], cov[0][1], cov[1][1]] == pytest.approx([s11, s12, s22], rel=1e-8)
    # the imputed value is the conditional mean at the MLE
    assert res.extra["imputed"][6][1] == pytest.approx(mu2 + s12 / s11 * (y1[6] - m1), rel=1e-9)
