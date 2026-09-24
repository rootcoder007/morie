"""Verification tests for hedderich8e71.

Hedderich, eq (8.71)-(8.72) -- the Cox-Snell and Nagelkerke measures. Every expected value is recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.hedderich8e71 import hedderich_chapter_8_equation_71


def test_the_three_pseudo_r2_measures():
    # (8.70)-(8.72)
    ll, ll0, n = -40.0, -60.0, 100
    res = hedderich_chapter_8_equation_71(ll, ll0, n)
    assert res["mcfadden"] == pytest.approx(1.0 - ll / ll0, rel=1e-12)
    cs = 1.0 - math.exp(2.0 * (ll0 - ll) / n)
    csmax = 1.0 - math.exp(2.0 * ll0 / n)
    assert res["coxsnell"] == pytest.approx(cs, rel=1e-12)
    assert res["coxsnell_max"] == pytest.approx(csmax, rel=1e-12)
    assert res["nagelkerke"] == pytest.approx(cs / csmax, rel=1e-12)


def test_a_model_no_better_than_the_null_scores_zero():
    res = hedderich_chapter_8_equation_71(-60.0, -60.0, 100)
    assert res["mcfadden"] == pytest.approx(0.0, abs=1e-12)
    assert res["coxsnell"] == pytest.approx(0.0, abs=1e-12)
    assert res["nagelkerke"] == pytest.approx(0.0, abs=1e-12)


def test_nagelkerke_rescales_cox_snell_to_reach_one():
    # as the model approaches a perfect fit, LL -> 0 and Nagelkerke -> 1
    res = hedderich_chapter_8_equation_71(-1e-9, -60.0, 100)
    assert res["nagelkerke"] == pytest.approx(1.0, abs=1e-6)
    assert res["coxsnell"] < 1.0


def test_requires_a_strictly_negative_null_log_likelihood():
    with pytest.raises(ValueError):
        hedderich_chapter_8_equation_71(-10.0, 0.0, 50)
