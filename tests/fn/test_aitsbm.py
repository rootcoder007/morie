"""Tests for aitsbm.compositional_simbias.

The generated placeholders here only checked that a dict came back. They
are replaced by the two properties Aitchison's argument actually turns
on: the variation array is invariant under subcomposition, and the raw
correlation is not.
"""

import numpy as np
import pytest

from morie.fn.aitsbm import compositional_simbias


def _composition(seed=42, n=200, D=5):
    """Strictly positive parts; the function closes them on entry."""
    return np.exp(np.random.default_rng(seed).normal(0, 1, (n, D)))


def test_variation_array_is_subcompositionally_coherent():
    """tau_ij = var(log(x_i/x_j)) cancels the closure constant exactly.

    Re-closing to a subcomposition multiplies every part of a row by one
    common factor, which drops out of the log-ratio. tau must therefore
    be identical in both, to floating-point error.
    """
    res = compositional_simbias(_composition(), idx=[0, 1, 2])
    assert res["tau_full"] == pytest.approx(res["tau_sub"], abs=1e-12)
    assert res["tau_delta"] == pytest.approx(0.0, abs=1e-12)


def test_raw_correlation_is_not_coherent():
    """The same pair correlates differently once other parts are dropped."""
    res = compositional_simbias(_composition(), idx=[0, 1, 2])
    assert res["delta"] == pytest.approx(res["rho_sub"] - res["rho_full"])
    assert abs(res["delta"]) > 0.05, "closure effect should be visible at D=5 -> 3"


def test_two_part_subcomposition_correlates_minus_one():
    """A 2-part subcomposition closes to (p, 1-p), so rho_sub == -1 exactly."""
    res = compositional_simbias(_composition(), idx=[0, 1])
    assert res["rho_sub"] == pytest.approx(-1.0, abs=1e-12)


def test_independent_parts_still_correlate_after_closure():
    """Pearson's point: closure manufactures correlation from independence."""
    res = compositional_simbias(_composition(seed=7, n=500, D=4), idx=[0, 1, 2])
    assert res["rho_full"] != pytest.approx(0.0, abs=1e-3)


def test_rejects_non_positive_parts():
    X = _composition()
    X[0, 0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        compositional_simbias(X, idx=[0, 1, 2])


def test_rejects_subcomposition_equal_to_composition():
    with pytest.raises(ValueError, match="fewer than"):
        compositional_simbias(_composition(D=4), idx=[0, 1, 2, 3])


def test_rejects_too_few_parts():
    with pytest.raises(ValueError, match="at least 3 parts"):
        compositional_simbias(_composition(D=2), idx=[0, 1])


def test_rejects_repeated_part():
    with pytest.raises(ValueError, match="repeat"):
        compositional_simbias(_composition(), idx=[0, 0, 1])
