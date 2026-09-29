"""Tests for km020.kamath_ch2_ssl_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km020 as mod


def test_km020_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km020_edge():
    import pytest

    from morie.fn.km020 import kamath_ch2_ssl_loss

    with pytest.raises(ValueError):
        kamath_ch2_ssl_loss([])


def test_weighted_ssl_loss_recomputed():
    r = mod.kamath_ch2_ssl_loss([0.7, 1.3, 2.0], [0.2, 0.5, 1.0])
    import pytest

    assert r["estimate"] == pytest.approx(0.7 * 0.2 + 1.3 * 0.5 + 2.0 * 1.0, rel=1e-15)
    assert r["components"] == [0.7 * 0.2, 1.3 * 0.5, 2.0]
