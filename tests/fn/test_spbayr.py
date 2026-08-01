"""spbayr -- Bayesian hierarchical disease mapping, Sec. 6.4."""

import numpy as np
import pytest


def _areas(n=10, seed=3):
    """A chain with two extra links, so degrees vary."""
    rs = np.random.RandomState(seed)
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    A[0, 4] = A[4, 0] = 1.0
    E = rs.uniform(8, 30, n)
    y = np.array([float(rs.poisson(e * np.exp(x)))
                  for e, x in zip(E, np.linspace(-0.3, 0.3, n))])
    return A, E, y

from morie.fn._schab_glmm import neighbour_structure
from morie.fn.spbayr import schabenberger_bayes_hierarchical


def test_exchangeable_prior_is_the_identity():
    """eq (6.102): aspatial."""
    A, E, y = _areas()
    r = schabenberger_bayes_hierarchical(y, E, A, spatial_prior="exchangeable")
    assert np.allclose(r["precision"], np.eye(len(y)))


def test_icar_prior_is_singular():
    """eq (6.104): improper, hence the pseudo-inverse elsewhere."""
    A, E, y = _areas()
    r = schabenberger_bayes_hierarchical(y, E, A, spatial_prior="icar")
    assert r["rank_deficiency_spatial"] == 1
    assert np.allclose(r["precision"], neighbour_structure(A))


def test_lcar_nests_both_extremes():
    """rho = 0 exchangeable, rho = 1 intrinsic."""
    A, E, y = _areas()
    lo = schabenberger_bayes_hierarchical(y, E, A, spatial_prior="lcar", rho=0.0)
    hi = schabenberger_bayes_hierarchical(y, E, A, spatial_prior="lcar", rho=1.0)
    assert np.allclose(lo["precision"], np.eye(len(y)))
    assert np.allclose(hi["precision"], neighbour_structure(A))


@pytest.mark.parametrize("kind,expected", [("I", 1), ("II", 1),
                                           ("III", 1), ("IV", 1)])
def test_constraints_equal_the_rank_deficiency(kind, expected):
    """eq (12): one constraint per unit of rank deficiency."""
    A, E, y = _areas()
    r = schabenberger_bayes_hierarchical(y, E, A, n_time=5,
                                         temporal_prior="rw1",
                                         interaction=kind)
    assert r["n_constraints"] == r["rank_deficiency"]


def test_type_one_interaction_needs_no_constraints():
    A, E, y = _areas()
    r = schabenberger_bayes_hierarchical(y, E, A, n_time=5,
                                         temporal_prior="rw1", interaction="I")
    assert r["rank_deficiency"] == 0
    assert r["n_constraints"] == 0


def test_type_four_rank_matches_the_table():
    """Table 1 with RW1: rank (I-1)(T-1)."""
    A, E, y = _areas()
    n, T = len(y), 5
    r = schabenberger_bayes_hierarchical(y, E, A, n_time=T,
                                         temporal_prior="rw1", interaction="IV")
    assert r["interaction_rank"] == (n - 1) * (T - 1)


def test_rw2_has_deficiency_two():
    A, E, y = _areas()
    r = schabenberger_bayes_hierarchical(y, E, A, n_time=6,
                                         temporal_prior="rw2")
    assert r["rank_deficiency_temporal"] == 2


def test_temporal_structure_requires_n_time():
    A, E, y = _areas()
    with pytest.raises(ValueError, match="n_time"):
        schabenberger_bayes_hierarchical(y, E, A, temporal_prior="rw1")


def test_unknown_priors_rejected():
    A, E, y = _areas()
    with pytest.raises(ValueError, match="spatial_prior"):
        schabenberger_bayes_hierarchical(y, E, A, spatial_prior="wishart")
    with pytest.raises(ValueError, match="temporal_prior"):
        schabenberger_bayes_hierarchical(y, E, A, n_time=5,
                                         temporal_prior="ar1")
