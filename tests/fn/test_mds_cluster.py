"""MDS / spatial-utility cluster: mmdsf, krust, isotr, smacf, nmdsf,
shrpd, procs, agrsc, eudst, rndut, stquo."""

from morie.fn import _array_core as np
import pytest

from morie.fn.agrmt import agreement_score
from morie.fn.agrsc import agreement_score_matrix
from morie.fn.eudst import euclidean_utility
from morie.fn.isotr import isotonic_regression_disparity
from morie.fn.krust import kruskal_stress
from morie.fn.mmdsf import metric_mds_torgerson
from morie.fn.nmdsf import nonmetric_mds
from morie.fn.procs import procrustes_rotation
from morie.fn.rndut import random_utility_model
from morie.fn.shrpd import shepard_diagram
from morie.fn.smacf import smacof_algorithm
from morie.fn.stquo import status_quo_spatial


def _planted(seed=0, n=12, k=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt((diff**2).sum(axis=2))
    return X, D


def _dists(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff**2).sum(axis=2))


def test_mmdsf_recovers_euclidean_distances():
    X, D = _planted(0)
    out = metric_mds_torgerson(D, n_dims=2)
    # a Euclidean D is reproduced exactly (up to rotation) in 2 dims
    assert _dists(out["coordinates"]) == pytest.approx(D, abs=1e-8)
    assert out["explained"] == pytest.approx(1.0, abs=1e-8)
    # eigenvalues beyond rank 2 vanish
    assert np.all(np.abs(out["eigenvalues"][2:]) < 1e-8)
    with pytest.raises(ValueError):
        metric_mds_torgerson(D + np.eye(D.shape[0]))  # nonzero diagonal
    with pytest.raises(ValueError):
        metric_mds_torgerson(D[:4, :5])


def test_krust_hand_values():
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert kruskal_stress(D, D)["stress"] == pytest.approx(0.0)
    Dc = np.array([[0.0, 2.0], [2.0, 0.0]])
    # S1 = sqrt((1-2)^2 / 2^2) = 0.5
    out = kruskal_stress(D, Dc)
    assert out["stress"] == pytest.approx(0.5)
    assert out["verbal"] == "unacceptable"
    with pytest.raises(ValueError):
        kruskal_stress(D, np.zeros((2, 2)))


def test_isotr_pav_hand_case():
    # classic PAV: [1, 3, 2, 4] with ranks 0..3 pools (3, 2) -> 2.5
    out = isotonic_regression_disparity([1.0, 3.0, 2.0, 4.0], [0, 1, 2, 3])
    assert out["disparities"] == pytest.approx([1.0, 2.5, 2.5, 4.0])
    assert np.all(np.diff(out["sorted_fit"]) >= -1e-12)
    # order of delta_rank, not input order, decides the sort
    out2 = isotonic_regression_disparity([4.0, 1.0], [1, 0])
    assert out2["disparities"] == pytest.approx([4.0, 1.0])
    with pytest.raises(ValueError):
        isotonic_regression_disparity([1.0], [0])


def test_smacf_majorization_never_increases_stress():
    X, D = _planted(1, n=10)
    out = smacof_algorithm(D, n_dims=2, max_iter=200)
    assert np.all(np.diff(out["stress_path"]) <= 1e-9)  # the guarantee
    assert out["stress"] < 1e-6  # planted config: near-perfect fit
    assert _dists(out["coordinates"]) == pytest.approx(D, abs=1e-3)
    # weighted variant runs and still decreases
    W = np.ones_like(D) - np.eye(D.shape[0])
    W[0, 1] = W[1, 0] = 5.0
    wout = smacof_algorithm(D, n_dims=2, weights=W, max_iter=100)
    assert np.all(np.diff(wout["stress_path"]) <= 1e-9)
    with pytest.raises(ValueError):
        smacof_algorithm(D, n_dims=2, weights=-W)


def test_nmdsf_invariant_to_monotone_distortion():
    # nonmetric MDS sees only ranks: cubing the distances must not stop
    # it from recovering a low-stress configuration.
    X, D = _planted(2, n=14)
    out = nonmetric_mds(D**3, n_dims=2, max_iter=200)
    assert out["stress"] < 0.05  # "good" on Kruskal's scale; measured ~0.01
    assert out["stress_path"][0] >= out["stress_path"][-1]
    with pytest.raises(ValueError):
        nonmetric_mds(D[:4, :5])


def test_shrpd_monotone_case():
    X, D = _planted(3, n=8)
    out = shepard_diagram(D**2, D)  # perfectly monotone relation
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert np.all(np.diff(out["monotone_fit"]) >= -1e-12)
    assert out["n_pairs"] == 8 * 7 // 2


def test_procs_undoes_rotation_and_reflection():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(9, 2))
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    F = np.array([[1.0, 0.0], [0.0, -1.0]])  # reflection
    Z = A @ (R @ F).T
    out = procrustes_rotation(A, Z)
    assert out["residual"] == pytest.approx(0.0, abs=1e-10)
    assert out["residual_before"] > 1.0
    assert out["reflection"] is True
    assert out["rotated"] == pytest.approx(A, abs=1e-10)
    with pytest.raises(ValueError):
        procrustes_rotation(A, Z[:5])


def test_agrsc_matches_agrmt():
    V = np.array([[1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, np.nan]])
    a = agreement_score_matrix(V)
    b = agreement_score(V)
    assert a["matrix"] == pytest.approx(b.value["agreement_matrix"])
    assert a["n"] == 3


def test_eudst_hand_values():
    out = euclidean_utility([0.0, 0.0], [3.0, 4.0])
    assert out["utility"] == pytest.approx(-25.0)
    assert out["distance"] == pytest.approx(5.0)
    m = euclidean_utility([[0.0], [1.0]], [[0.0], [2.0]])
    assert np.asarray(m["utility"]) == pytest.approx(np.array([[0.0, -4.0], [-1.0, -1.0]]))
    with pytest.raises(ValueError):
        euclidean_utility([0.0, 0.0], [1.0, 2.0, 3.0])


def test_rndut_gumbel_is_softmax_and_probit_is_symmetric():
    V = np.array([1.0, 0.0, -1.0])
    out = random_utility_model(V, "gumbel")
    ez = np.exp(V)
    assert out["probabilities"] == pytest.approx(ez / ez.sum())
    assert out["chosen"] == 0
    # probit: equal utilities -> equal probabilities (measured +/- 0.01)
    p = random_utility_model([0.0, 0.0], "normal", n_draws=40000, seed=0)
    assert p["probabilities"][0] == pytest.approx(0.5, abs=0.02)
    with pytest.raises(ValueError):
        random_utility_model(V, "cauchy")


def test_stquo_median_voter():
    # 1-D: the alternative closer to the median voter wins
    ideal = [0.0, 1.0, 2.0, 3.0, 10.0]  # median 2
    out = status_quo_spatial(ideal, status_quo=0.0, proposal=2.5)
    assert out["passes"] is True  # 2.5 is closer than 0.0 for voters at 2, 3, 10
    assert out["votes_for"] == 3
    lose = status_quo_spatial(ideal, status_quo=2.0, proposal=8.0)
    assert lose["passes"] is False
    tie = status_quo_spatial([0.0, 2.0], status_quo=1.0 - 1e-12, proposal=1.0 + 1e-12)
    assert tie["indifferent"] >= 0  # exact-tie bookkeeping exists
    with pytest.raises(ValueError):
        status_quo_spatial(ideal, [0.0, 0.0], 1.0)
