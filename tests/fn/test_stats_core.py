"""Equivalence: _stats_core vs frozen scipy.stats anchors on the used
surface (versions recorded in oracle_anchors.json; scipy itself is not
imported)."""

import json
import pathlib

import pytest

from morie.fn import _stats_core as ms

S = json.loads(pathlib.Path(__file__).with_name(
    "oracle_anchors.json").read_text())["stats"]


def eq(a, b, tol=1e-9):
    assert a == pytest.approx(b, rel=tol, abs=tol)


class TestNorm:
    def test_pdf_cdf_ppf(self):
        w = S["norm"]
        for x in (-3.0, -0.5, 0.0, 1.2, 4.0):
            eq(ms.norm.pdf(x), w["pdf"][str(x)], 1e-12)
            eq(ms.norm.cdf(x), w["cdf"][str(x)], 1e-12)
            eq(ms.norm.sf(x), w["sf"][str(x)], 1e-12)
        for q in (0.001, 0.025, 0.5, 0.975, 0.999):
            eq(ms.norm.ppf(q), w["ppf"][str(q)], 1e-12)
        eq(ms.norm.ppf(0.9, loc=2, scale=3), w["ppf_loc"], 1e-12)
        eq(ms.norm.cdf(1.0, 2.0, 3.0), w["cdf_loc"], 1e-12)


class TestChi2:
    def test_all(self):
        for df in (1, 2, 5, 30):
            w = S["chi2"][str(df)]
            for x in (0.5, 2.0, 10.0, 40.0):
                eq(ms.chi2.pdf(x, df), w["pdf"][str(x)], 1e-10)
                eq(ms.chi2.cdf(x, df), w["cdf"][str(x)], 1e-10)
                eq(ms.chi2.sf(x, df), w["sf"][str(x)], 1e-8)
            for q in (0.05, 0.5, 0.95, 0.995):
                eq(ms.chi2.ppf(q, df), w["ppf"][str(q)], 1e-8)


class TestT:
    def test_all(self):
        for df in (1, 5, 30, 200):
            w = S["t"][str(df)]
            for x in (-3.0, -0.5, 0.0, 2.0):
                eq(ms.t.pdf(x, df), w["pdf"][str(x)], 1e-10)
                eq(ms.t.cdf(x, df), w["cdf"][str(x)], 1e-10)
            for q in (0.025, 0.5, 0.9, 0.975):
                eq(ms.t.ppf(q, df), w["ppf"][str(q)], 1e-8)


class TestF:
    def test_all(self):
        for d1, d2 in ((1, 10), (5, 20), (10, 2)):
            w = S["f"]["%d_%d" % (d1, d2)]
            for x in (0.3, 1.0, 3.5):
                eq(ms.f.pdf(x, d1, d2), w["pdf"][str(x)], 1e-10)
                eq(ms.f.cdf(x, d1, d2), w["cdf"][str(x)], 1e-10)
            for q in (0.05, 0.5, 0.95):
                eq(ms.f.ppf(q, d1, d2), w["ppf"][str(q)], 1e-7)


class TestGammaBeta:
    def test_gamma(self):
        for a in (0.5, 2.0, 7.5):
            w = S["gamma"][str(a)]
            for x in (0.2, 1.0, 6.0):
                eq(ms.gamma.pdf(x, a), w["pdf"][str(x)], 1e-10)
                eq(ms.gamma.cdf(x, a), w["cdf"][str(x)], 1e-10)
            eq(ms.gamma.ppf(0.9, a), w["ppf09"], 1e-8)

    def test_beta(self):
        for a, b in ((2, 3), (0.5, 0.5), (8, 14)):
            w = S["beta"]["%s_%s" % (a, b)]
            for x in (0.1, 0.5, 0.9):
                eq(ms.beta.pdf(x, a, b), w["pdf"][str(x)], 1e-10)
                eq(ms.beta.cdf(x, a, b), w["cdf"][str(x)], 1e-10)
            eq(ms.beta.ppf(0.75, a, b), w["ppf075"], 1e-8)


class TestDiscrete:
    def test_binom(self):
        for k in (0, 3, 7, 10):
            w = S["binom"][str(k)]
            eq(ms.binom.pmf(k, 10, 0.3), w["pmf"], 1e-12)
            eq(ms.binom.cdf(k, 10, 0.3), w["cdf"], 1e-10)
            eq(ms.binom.sf(k, 10, 0.3), w["sf"], 1e-9)

    def test_poisson(self):
        for k in (0, 2, 5, 12):
            w = S["poisson"][str(k)]
            eq(ms.poisson.pmf(k, 3.5), w["pmf"], 1e-12)
            eq(ms.poisson.cdf(k, 3.5), w["cdf"], 1e-10)


class TestSimple:
    def test_uniform_expon(self):
        w = S["simple"]
        eq(ms.uniform.cdf(0.3), w["uniform_cdf"], 1e-14)
        eq(ms.uniform.ppf(0.7, 2, 4), w["uniform_ppf"], 1e-14)
        eq(ms.expon.cdf(1.5), w["expon_cdf"], 1e-14)
        eq(ms.expon.ppf(0.9), w["expon_ppf"], 1e-12)

    def test_helpers(self):
        w = S["simple"]
        x = [2.0, 4.0, 4.0, 5.0, 7.0]
        eq(ms.sem(x), w["sem"], 1e-12)
        z = ms.zscore(x)
        for a, b in zip(z.tolist(), w["zscore"]):
            eq(a, b, 1e-12)

    def test_frozen(self):
        fr = ms.norm(2.0, 3.0)
        eq(fr.cdf(4.0), S["norm"]["frozen_cdf"], 1e-12)
        fc = ms.chi2(5)
        eq(fc.ppf(0.9), S["chi2_frozen_ppf"], 1e-8)
