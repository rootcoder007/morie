"""Tests for plrgrf. Full anchor: ledger/wave3/anchor_grf_family.py."""
import pytest
from morie.fn import _s03core as k
from morie.fn.plrgrf import local_centering, partial_linear_grf
from ._grf_fixture import confounded


@pytest.fixture(scope="module")
def d():
    return confounded(700, 21, m_scale=3.0)


def test_local_centering_recovers_the_cate_and_skipping_it_does_not(d):
    """The whole point: without residualising, the forest splits on the
    confounding surface m(X) instead of on tau."""
    grid = [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    truth = [0.5 + g[1] for g in grid]
    c = partial_linear_grf(d["y"], d["W"], d["X"], at=grid, n_trees=120,
                           min_leaf=5, seed=1)
    u = partial_linear_grf(d["y"], d["W"], d["X"], at=grid, n_trees=120,
                           min_leaf=5, seed=1, center=False)
    ec = k.mean([abs(c["tau"][q] - truth[q]) for q in range(3)])
    eu = k.mean([abs(u["tau"][q] - truth[q]) for q in range(3)])
    assert ec < 0.5
    assert eu > 2.0 * ec
    assert c["tau"][0] < c["tau"][1] < c["tau"][2]


def test_the_cross_fitted_propensity_tracks_the_true_one(d):
    m, e = local_centering(d["y"], d["W"], d["X"], n_folds=5,
                           n_trees=60, min_leaf=5, seed=1)
    assert k.corr(e, d["e"]) > 0.5
    assert len(m) == len(e) == d["n"]


def test_argument_checks(d):
    with pytest.raises(ValueError):
        partial_linear_grf(d["y"][:-1], d["W"], d["X"])
    with pytest.raises(ValueError):
        partial_linear_grf(d["y"], d["W"], d["X"][:-1])
    with pytest.raises(ValueError):
        partial_linear_grf(d["y"][:20], d["W"][:20], d["X"][:20])
