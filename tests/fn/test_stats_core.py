"""Equivalence: _stats_core vs scipy.stats on the used surface."""

import pytest

scipy_stats = pytest.importorskip("scipy.stats")

from morie.fn import _stats_core as ms  # noqa: E402


def eq(a, b, tol=1e-9):
    assert a == pytest.approx(b, rel=tol, abs=tol)


class TestNorm:
    def test_pdf_cdf_ppf(self):
        for x in (-3.0, -0.5, 0.0, 1.2, 4.0):
            eq(ms.norm.pdf(x), scipy_stats.norm.pdf(x), 1e-12)
            eq(ms.norm.cdf(x), scipy_stats.norm.cdf(x), 1e-12)
            eq(ms.norm.sf(x), scipy_stats.norm.sf(x), 1e-12)
        for q in (0.001, 0.025, 0.5, 0.975, 0.999):
            eq(ms.norm.ppf(q), scipy_stats.norm.ppf(q), 1e-12)
        eq(ms.norm.ppf(0.9, loc=2, scale=3),
           scipy_stats.norm.ppf(0.9, loc=2, scale=3), 1e-12)
        eq(ms.norm.cdf(1.0, 2.0, 3.0),
           scipy_stats.norm.cdf(1.0, 2.0, 3.0), 1e-12)


class TestChi2:
    def test_all(self):
        for df in (1, 2, 5, 30):
            for x in (0.5, 2.0, 10.0, 40.0):
                eq(ms.chi2.pdf(x, df), scipy_stats.chi2.pdf(x, df), 1e-10)
                eq(ms.chi2.cdf(x, df), scipy_stats.chi2.cdf(x, df), 1e-10)
                eq(ms.chi2.sf(x, df), scipy_stats.chi2.sf(x, df), 1e-8)
            for q in (0.05, 0.5, 0.95, 0.995):
                eq(ms.chi2.ppf(q, df), scipy_stats.chi2.ppf(q, df), 1e-8)


class TestT:
    def test_all(self):
        for df in (1, 5, 30, 200):
            for x in (-3.0, -0.5, 0.0, 2.0):
                eq(ms.t.pdf(x, df), scipy_stats.t.pdf(x, df), 1e-10)
                eq(ms.t.cdf(x, df), scipy_stats.t.cdf(x, df), 1e-10)
            for q in (0.025, 0.5, 0.9, 0.975):
                eq(ms.t.ppf(q, df), scipy_stats.t.ppf(q, df), 1e-8)


class TestF:
    def test_all(self):
        for d1, d2 in ((1, 10), (5, 20), (10, 2)):
            for x in (0.3, 1.0, 3.5):
                eq(ms.f.pdf(x, d1, d2), scipy_stats.f.pdf(x, d1, d2), 1e-10)
                eq(ms.f.cdf(x, d1, d2), scipy_stats.f.cdf(x, d1, d2), 1e-10)
            for q in (0.05, 0.5, 0.95):
                eq(ms.f.ppf(q, d1, d2), scipy_stats.f.ppf(q, d1, d2), 1e-7)


class TestGammaBeta:
    def test_gamma(self):
        for a in (0.5, 2.0, 7.5):
            for x in (0.2, 1.0, 6.0):
                eq(ms.gamma.pdf(x, a), scipy_stats.gamma.pdf(x, a), 1e-10)
                eq(ms.gamma.cdf(x, a), scipy_stats.gamma.cdf(x, a), 1e-10)
            eq(ms.gamma.ppf(0.9, a), scipy_stats.gamma.ppf(0.9, a), 1e-8)

    def test_beta(self):
        for a, b in ((2, 3), (0.5, 0.5), (8, 14)):
            for x in (0.1, 0.5, 0.9):
                eq(ms.beta.pdf(x, a, b), scipy_stats.beta.pdf(x, a, b),
                   1e-10)
                eq(ms.beta.cdf(x, a, b), scipy_stats.beta.cdf(x, a, b),
                   1e-10)
            eq(ms.beta.ppf(0.75, a, b), scipy_stats.beta.ppf(0.75, a, b),
               1e-8)


class TestDiscrete:
    def test_binom(self):
        for k in (0, 3, 7, 10):
            eq(ms.binom.pmf(k, 10, 0.3), scipy_stats.binom.pmf(k, 10, 0.3),
               1e-12)
            eq(ms.binom.cdf(k, 10, 0.3), scipy_stats.binom.cdf(k, 10, 0.3),
               1e-10)
            eq(ms.binom.sf(k, 10, 0.3), scipy_stats.binom.sf(k, 10, 0.3),
               1e-9)

    def test_poisson(self):
        for k in (0, 2, 5, 12):
            eq(ms.poisson.pmf(k, 3.5), scipy_stats.poisson.pmf(k, 3.5),
               1e-12)
            eq(ms.poisson.cdf(k, 3.5), scipy_stats.poisson.cdf(k, 3.5),
               1e-10)


class TestSimple:
    def test_uniform_expon(self):
        eq(ms.uniform.cdf(0.3), scipy_stats.uniform.cdf(0.3), 1e-14)
        eq(ms.uniform.ppf(0.7, 2, 4), scipy_stats.uniform.ppf(0.7, 2, 4),
           1e-14)
        eq(ms.expon.cdf(1.5), scipy_stats.expon.cdf(1.5), 1e-14)
        eq(ms.expon.ppf(0.9), scipy_stats.expon.ppf(0.9), 1e-12)

    def test_helpers(self):
        x = [2.0, 4.0, 4.0, 5.0, 7.0]
        eq(ms.sem(x), float(scipy_stats.sem(x)), 1e-12)
        z = ms.zscore(x)
        zs = scipy_stats.zscore(x)
        for a, b in zip(z.tolist(), zs.tolist()):
            eq(a, b, 1e-12)

    def test_frozen(self):
        fr = ms.norm(2.0, 3.0)
        eq(fr.cdf(4.0), scipy_stats.norm(2.0, 3.0).cdf(4.0), 1e-12)
        fc = ms.chi2(5)
        eq(fc.ppf(0.9), scipy_stats.chi2(5).ppf(0.9), 1e-8)
