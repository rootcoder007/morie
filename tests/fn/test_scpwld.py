"""Tests for morie.fn.scpwld: Wald chi-square test of the spatial parameter."""

import math

import pytest

from morie.fn.scpwld import scpwld, scpwld_fn


def test_statistic_and_pvalue():
    for rho, se, r0 in ((0.3, 0.1, 0.0), (-0.12, 0.2, 0.0), (0.5, 0.15, 0.2)):
        r = scpwld(rho, se, r0)
        z = (rho - r0) / se
        assert r.statistic == pytest.approx(z * z, abs=1e-12)
        # chi-square(1) survival = 2 * (1 - Phi(|z|))
        assert r.p_value == pytest.approx(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))), abs=1e-12)
        assert r.df == 1


def test_bad_se_and_alias():
    with pytest.raises(ValueError):
        scpwld(0.3, 0.0)
    assert scpwld_fn is scpwld
