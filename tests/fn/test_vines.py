"""vines: vine-copula dependence structure.

Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas* (Springer) --
in the library as 13 page-range volumes.
"""

import numpy as np
import pytest

from morie.fn.vines import vine_copula as vc


def test_vines_correlation_matrix_is_valid():
    """Symmetric, unit diagonal, entries in [-1, 1]."""
    rng = np.random.default_rng(3601)
    R = np.asarray(vc(rng.standard_normal((300, 4)))["R"])
    assert R.shape == (4, 4)
    assert np.allclose(R, R.T)
    assert np.allclose(np.diag(R), 1.0)
    assert np.all(np.abs(R) <= 1.0 + 1e-9)


def test_vines_recovers_a_planted_dependence():
    """Two strongly coupled columns must show a large pairwise entry."""
    rng = np.random.default_rng(3607)
    a = rng.standard_normal(500)
    x = np.column_stack([a, 0.9 * a + 0.436 * rng.standard_normal(500),
                         rng.standard_normal(500)])
    R = np.asarray(vc(x)["R"])
    assert abs(R[0, 1]) > 0.7
    assert abs(R[0, 2]) < 0.3


def test_vines_independent_columns_give_a_near_identity_matrix():
    rng = np.random.default_rng(3613)
    R = np.asarray(vc(rng.standard_normal((3000, 3)))["R"])
    off = R - np.eye(3)
    assert np.max(np.abs(off)) < 0.1


def test_vines_partial_correlations_are_the_vine_parameterisation():
    """A vine is built from partial correlations, so they must be present,
    finite, and inside [-1, 1] -- that is what distinguishes this from a
    plain correlation matrix."""
    rng = np.random.default_rng(3617)
    pc = np.asarray(vc(rng.standard_normal((400, 4)))["partial_corr"])
    assert pc.size > 0
    assert np.all(np.isfinite(pc))
    assert np.all(np.abs(pc) <= 1.0 + 1e-9)


def test_vines_conditional_independence_shows_in_the_partial_correlation():
    """The point of a vine: x3 depends on x1 only through x2, so the partial
    correlation of (x1, x3) given x2 must be far smaller than their marginal
    correlation."""
    rng = np.random.default_rng(3623)
    n = 4000
    x1 = rng.standard_normal(n)
    x2 = 0.9 * x1 + 0.436 * rng.standard_normal(n)
    x3 = 0.9 * x2 + 0.436 * rng.standard_normal(n)
    R = np.asarray(vc(np.column_stack([x1, x2, x3]))["R"])
    marginal = abs(R[0, 2])
    # partial corr of 1,3 given 2, computed independently from R
    part = abs((R[0, 2] - R[0, 1] * R[1, 2]) /
               np.sqrt((1 - R[0, 1] ** 2) * (1 - R[1, 2] ** 2)))
    assert marginal > 0.6
    assert part < 0.15


def test_vines_reports_shape_and_finite_loglik():
    rng = np.random.default_rng(3629)
    r = vc(rng.standard_normal((250, 5)))
    assert r["n"] == 250 and r["d"] == 5
    assert np.isfinite(r["loglik"])
