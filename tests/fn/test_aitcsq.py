"""Tests for aitcsq.compositional_chisq.

Checked against scipy.stats.chi2_contingency, which computes the same
Pearson statistic by the ordinary (o - e)^2 / e route, and against the
identities the correspondence-analysis construction guarantees.
"""

import numpy as np
import pytest
from scipy import stats

from morie.fn.aitcsq import compositional_chisq


def _table(seed=0, I=5, J=4, lam=40):
    return np.random.default_rng(seed).poisson(lam, (I, J)).astype(float)


def test_matches_scipy_chi2_contingency():
    """The CA route and the (o - e)^2 / e route are the same number."""
    for seed in range(5):
        X = _table(seed=seed)
        got = compositional_chisq(X)
        ref = stats.chi2_contingency(X, correction=False)
        assert got["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
        assert got["df"] == ref.dof
        assert got["p_value"] == pytest.approx(ref.pvalue, rel=1e-10)


def test_chi_square_is_n_times_inertia():
    X = _table(seed=1)
    res = compositional_chisq(X)
    assert res["statistic"] == pytest.approx(res["n"] * res["inertia"], rel=1e-12)


def test_total_inertia_is_the_sum_of_principal_inertias():
    """SVD decomposes ||S||_F^2 into the squared singular values."""
    res = compositional_chisq(_table(seed=2))
    assert float(res["principal_inertias"].sum()) == pytest.approx(res["inertia"], rel=1e-12)


def test_number_of_principal_inertias_is_the_table_rank():
    """min(I, J) singular values, of which at most min(I, J) - 1 are non-zero.

    One dimension is lost because the row profiles all lie in a simplex:
    the trivial axis carries the masses, not the association.
    """
    res = compositional_chisq(_table(seed=3, I=5, J=4))
    sv = res["singular_values"]
    assert sv.size == 4
    assert np.sum(sv > 1e-10) <= 3


def test_inertia_is_invariant_to_row_scaling_but_chi_square_is_not():
    """Inertia depends only on profiles; the statistic scales with n."""
    X = _table(seed=4)
    scaled = X * np.array([1.0, 2.0, 3.0, 4.0, 5.0])[:, None]
    a, b = compositional_chisq(X), compositional_chisq(scaled)
    assert b["inertia"] != pytest.approx(a["inertia"], rel=1e-6), "row scaling changes the profiles' weights"
    # Closing every row to the same total leaves the profiles untouched,
    # and then the inertia must not move at all.
    closed = X / X.sum(axis=1, keepdims=True)
    assert compositional_chisq(closed)["inertia"] == pytest.approx(
        compositional_chisq(closed * 7.0)["inertia"], rel=1e-12
    )


def test_independent_table_has_near_zero_inertia():
    """A rank-one table r c' is exactly independent, so S is zero."""
    r = np.array([10.0, 20.0, 30.0])
    c = np.array([0.2, 0.3, 0.5])
    assert compositional_chisq(np.outer(r, c))["inertia"] == pytest.approx(0.0, abs=1e-24)


def test_n_override_rescales_the_statistic():
    X = X = _table(seed=5)
    res = compositional_chisq(X, n=1000.0)
    assert res["n"] == 1000.0
    assert res["statistic"] == pytest.approx(1000.0 * res["inertia"], rel=1e-12)


def test_masses_sum_to_one():
    res = compositional_chisq(_table(seed=6))
    assert float(res["row_masses"].sum()) == pytest.approx(1.0)
    assert float(res["col_masses"].sum()) == pytest.approx(1.0)


def test_supplied_cdf_replaces_the_asymptotic_null():
    X = _table(seed=7)
    res = compositional_chisq(X, cdf=stats.chi2(12).cdf)
    assert res["p_value"] == pytest.approx(compositional_chisq(X)["p_value"], rel=1e-9)


def test_validates_inputs():
    with pytest.raises(ValueError, match="non-negative"):
        compositional_chisq(np.random.default_rng(0).normal(0, 1, (10, 4)))
    with pytest.raises(ValueError, match="at least a 2x2"):
        compositional_chisq(np.ones((1, 4)))
    with pytest.raises(ValueError, match="must be finite"):
        compositional_chisq(np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError, match="sums to zero"):
        compositional_chisq(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="positive mass"):
        compositional_chisq(np.array([[1.0, 2.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="n must be positive"):
        compositional_chisq(_table(seed=8), n=0)
