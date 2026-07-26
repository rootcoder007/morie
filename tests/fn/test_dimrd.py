"""dimrd: Cattell scree / Kaiser dimensionality test (Kaiser 1960; Armstrong)."""

import numpy as np
import pytest

from morie.fn.dimrd import dimensionality_test as dt


def test_dimrd_recovers_a_planted_two_factor_structure():
    """Six variables built from two orthogonal latent factors: Kaiser's
    lambda > 1 rule must find exactly 2 dimensions."""
    rng = np.random.default_rng(29)
    n = 4000
    f1, f2 = rng.standard_normal(n), rng.standard_normal(n)
    X = np.column_stack(
        [f1 + 0.2 * rng.standard_normal(n) for _ in range(3)]
        + [f2 + 0.2 * rng.standard_normal(n) for _ in range(3)]
    )
    assert dt(X)["n_dims"] == 2


def test_dimrd_finds_one_dimension_when_everything_loads_on_one_factor():
    rng = np.random.default_rng(31)
    n = 4000
    f = rng.standard_normal(n)
    X = np.column_stack([f + 0.2 * rng.standard_normal(n) for _ in range(5)])
    assert dt(X)["n_dims"] == 1


def test_dimrd_eigenvalues_are_sorted_and_sum_to_the_trace():
    """For a correlation matrix of p variables the eigenvalues sum to p."""
    rng = np.random.default_rng(37)
    X = rng.standard_normal((500, 6))
    ev = np.asarray(dt(X)["eigenvalues"])
    assert np.all(np.diff(ev) <= 1e-9), "eigenvalues must be descending"
    assert ev.sum() == pytest.approx(6.0, rel=1e-9)


def test_dimrd_threshold_is_the_kaiser_rule_and_is_tunable():
    """Raising the cutoff cannot increase the dimension count."""
    rng = np.random.default_rng(41)
    n = 3000
    f1, f2 = rng.standard_normal(n), rng.standard_normal(n)
    X = np.column_stack(
        [f1 + 0.3 * rng.standard_normal(n) for _ in range(3)]
        + [f2 + 0.3 * rng.standard_normal(n) for _ in range(3)]
    )
    counts = [dt(X, threshold=t)["n_dims"] for t in (0.5, 1.0, 2.0, 3.0)]
    assert counts == sorted(counts, reverse=True)
    assert dt(X, threshold=1.0)["threshold"] == pytest.approx(1.0)


def test_dimrd_scree_gap_is_the_drop_after_the_retained_dimensions():
    """The gap must be positive where there is a genuine elbow."""
    rng = np.random.default_rng(43)
    n = 3000
    f = rng.standard_normal(n)
    X = np.column_stack([f + 0.1 * rng.standard_normal(n) for _ in range(4)])
    r = dt(X)
    assert r["n_dims"] == 1
    assert r["scree_gap"] > 0.0


def test_dimrd_accepts_a_symmetric_matrix_directly():
    """A square symmetric input is used as-is rather than correlated again."""
    S = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ev = np.asarray(dt(S)["eigenvalues"])
    assert ev == pytest.approx([1.9, 1.0, 0.1], abs=1e-9)
