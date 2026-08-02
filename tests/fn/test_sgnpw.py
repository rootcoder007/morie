"""Tests for sgnpw.sign_test_power."""

from morie.fn import _array_core as np
import pytest

from morie.fn.sgnpw import sign_test_power


def test_sgnpw_power_rises_with_the_alternative_probability():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    p6 = float(sign_test_power(x, p_alt=0.6)["statistic"])
    p9 = float(sign_test_power(x, p_alt=0.9)["statistic"])
    assert 0.0 <= p6 <= p9 <= 1.0
    assert p9 > 0.9


def test_sgnpw_null_probability_gives_alpha():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50)
    r = sign_test_power(x, p_alt=0.5, alpha=0.05)
    assert float(r["statistic"]) <= 0.07  # exact binomial test is conservative


def test_sgnpw_respects_alpha():
    rng = np.random.default_rng(2)
    x = rng.normal(size=40)
    strict = float(sign_test_power(x, p_alt=0.7, alpha=0.01)["statistic"])
    loose = float(sign_test_power(x, p_alt=0.7, alpha=0.10)["statistic"])
    assert strict <= loose
