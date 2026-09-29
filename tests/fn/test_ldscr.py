"""Tests for morie.fn.ldscr -- LD score regression."""

import pytest

from morie.fn import _array_core as np
from morie.fn.ldscr import ldscr


class TestLdscr:
    def test_null_h2(self):
        rng = np.random.default_rng(42)
        p = 200
        chi2 = rng.chisquare(1, size=p)
        ld = rng.uniform(1, 10, size=p)
        res = ldscr(chi2, ld, n_gwas=10000)
        assert abs(res.statistic) < 0.5

    def test_positive_h2(self):
        rng = np.random.default_rng(42)
        p = 200
        ld = rng.uniform(1, 20, size=p)
        chi2 = 1.0 + 0.5 * ld + rng.standard_normal(p) * 0.5
        chi2 = np.maximum(chi2, 0.01)
        res = ldscr(chi2, ld, n_gwas=10000)
        assert res.statistic > 0

    def test_fixed_intercept(self):
        rng = np.random.default_rng(42)
        p = 100
        ld = rng.uniform(1, 10, size=p)
        chi2 = 1.0 + rng.chisquare(1, size=p)
        res = ldscr(chi2, ld, n_gwas=5000, intercept_fixed=1.0)
        assert res.extra["intercept"] == 1.0

    def test_lambda_gc(self):
        chi2 = np.ones(100)
        ld = np.ones(100)
        res = ldscr(chi2, ld, n_gwas=1000)
        assert res.extra["lambda_gc"] > 0

    def test_mismatched_length(self):
        with pytest.raises(ValueError):
            ldscr(np.ones(10), np.ones(5))


def test_ld_score_regression_recomputed():
    chi2 = [1.2, 1.5, 2.1, 1.1, 2.8, 1.9]
    ld = [2.0, 4.0, 6.0, 1.5, 9.0, 5.0]
    n, M, N = 6, 6, 5000
    mx, my = sum(ld) / n, sum(chi2) / n
    b = sum((a - mx) * (c - my) for a, c in zip(ld, chi2)) / sum((a - mx) ** 2 for a in ld)
    res = ldscr(chi2, ld, n_gwas=N)
    assert res.extra["slope"] == pytest.approx(b, rel=1e-10)
    assert res.extra["intercept"] == pytest.approx(my - b * mx, rel=1e-10)
    assert res.statistic == pytest.approx(b * M / N, rel=1e-10)
    fixed = ldscr(chi2, ld, n_gwas=N, intercept_fixed=1.0)
    assert fixed.extra["slope"] == pytest.approx(
        sum(a * (c - 1) for a, c in zip(ld, chi2)) / sum(a * a for a in ld), rel=1e-12
    )
