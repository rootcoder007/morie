"""Tests for morie.fn.defuz -- defuzzification (Ross 2010, Fuzzy Logic with
Engineering Applications, 3rd ed.)."""

import numpy as np
import pytest

from morie.fn.defuz import defuz


def _gauss(x, mu, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def test_defuz_centroid_of_a_symmetric_membership_is_its_centre():
    """A Gaussian centred at 5 defuzzifies to 5 by symmetry -- analytic truth."""
    x = np.linspace(0, 10, 501)
    assert defuz(x=x, mf=_gauss(x, 5.0)).value == pytest.approx(5.0, abs=1e-6)


def test_defuz_centroid_follows_an_asymmetric_membership():
    """Skewing the membership toward 8 must move the centroid toward 8.

    A 1:3 mixture of unit Gaussians at 3 and 8 has its mass centre at
    (1*3 + 3*8)/4 = 6.75; the small shortfall is the left tail clipped by the
    x >= 0 boundary."""
    x = np.linspace(0, 10, 501)
    mf = _gauss(x, 3.0) + 3.0 * _gauss(x, 8.0)
    assert defuz(x=x, mf=mf).value == pytest.approx(6.75, abs=0.1)


def test_defuz_all_five_methods_agree_on_a_symmetric_membership():
    """centroid, bisector, mom, som and lom coincide only when the membership
    is symmetric and single-peaked. That is the case that pins them together;
    any method computing the wrong quantity shows up as a disagreement."""
    x = np.linspace(0, 10, 501)
    mf = _gauss(x, 5.0)
    for method in ("centroid", "bisector", "mom", "som", "lom"):
        assert defuz(x=x, mf=mf, method=method).value == pytest.approx(5.0, abs=0.05)


def test_defuz_som_mom_lom_bracket_a_membership_plateau():
    """On a flat top from 4 to 6: som = 4, lom = 6, mom = 5."""
    x = np.linspace(0, 10, 1001)
    mf = np.where((x >= 4.0) & (x <= 6.0), 1.0, 0.0)
    assert defuz(x=x, mf=mf, method="som").value == pytest.approx(4.0, abs=1e-2)
    assert defuz(x=x, mf=mf, method="lom").value == pytest.approx(6.0, abs=1e-2)
    assert defuz(x=x, mf=mf, method="mom").value == pytest.approx(5.0, abs=1e-2)


def test_defuz_bisector_splits_the_area_in_half():
    """By definition the bisector has equal area either side of it."""
    x = np.linspace(0, 10, 2001)
    mf = _gauss(x, 3.0) + 2.0 * _gauss(x, 7.0)
    z = defuz(x=x, mf=mf, method="bisector").value
    left = np.trapezoid(np.where(x <= z, mf, 0.0), x)
    right = np.trapezoid(np.where(x >= z, mf, 0.0), x)
    assert left == pytest.approx(right, rel=0.02)


def test_defuz_reports_total_area_and_peak():
    x = np.linspace(0, 10, 501)
    r = defuz(x=x, mf=_gauss(x, 5.0))
    assert r.extra["max_membership"] == pytest.approx(1.0)
    # A unit Gaussian integrates to sqrt(2*pi) well inside +/-5 sigma.
    assert r.extra["total_area"] == pytest.approx(np.sqrt(2 * np.pi), rel=1e-3)


def test_defuz_rejects_a_zero_membership_and_bad_method():
    x = np.linspace(0, 10, 50)
    with pytest.raises(ValueError, match="sums to zero"):
        defuz(x=x, mf=np.zeros_like(x))
    with pytest.raises(ValueError, match="Unknown method"):
        defuz(x=x, mf=_gauss(x, 5.0), method="nope")


def test_defuz_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        defuz(x=np.linspace(0, 1, 10), mf=np.ones(9))


def test_cheatsheet():
    from morie.fn.defuz import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
