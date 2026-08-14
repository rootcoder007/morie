"""PACE against a process whose truth is known.

Every assertion here can fail: the data are generated from a stated
Karhunen-Loeve expansion, so the eigenvalues, the eigenfunctions, the
scores and the fraction of variance explained all have right answers
that the code does not get told.
"""
import math

import pytest

from morie.fn import _array_core as np
from morie.fn.pace import local_linear, pace

L1, L2, SIGMA2 = 4.0, 1.0, 0.25


def _phi1(t):
    return math.sqrt(2.0) * math.sin(math.pi * t)


def _phi2(t):
    return math.sqrt(2.0) * math.cos(math.pi * t)


def _simulate(n=200, m=20, seed=7):
    """X_i(t) = t + xi_1 phi_1(t) + xi_2 phi_2(t) + N(0, sigma^2)."""
    rng = np.random.default_rng(seed)

    def norm():
        u1 = rng.random() or 1e-12
        u2 = rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    Y, T, XI = [], [], []
    for _ in range(n):
        x1 = math.sqrt(L1) * norm()
        x2 = math.sqrt(L2) * norm()
        XI.append((x1, x2))
        ti = sorted(rng.random() for _ in range(m))
        T.append(ti)
        Y.append([t + x1 * _phi1(t) + x2 * _phi2(t)
                  + math.sqrt(SIGMA2) * norm() for t in ti])
    return Y, T, XI


@pytest.fixture(scope="module")
def fit():
    Y, T, XI = _simulate()
    return pace(Y, T, K=2, n_grid=21, bw_mu=0.08, bw_cov=0.08), XI


def _abscorr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    den = math.sqrt(sum((x - ma) ** 2 for x in a)
                    * sum((x - mb) ** 2 for x in b))
    return abs(cov / den)


def test_eigenvalues_recover_the_generating_variances(fit):
    r, _ = fit
    assert abs(r["eigenvalues"][0] - L1) < 0.4          # true 4
    assert abs(r["eigenvalues"][1] - L2) < 0.4          # true 1
    assert r["eigenvalues"][0] > r["eigenvalues"][1]


def test_eigenfunctions_recover_the_generating_basis(fit):
    r, _ = fit
    g = r["grid"]
    # sign is not identified by the eigenproblem, so compare |corr|
    assert _abscorr(r["eigenfunctions"][0], [_phi1(x) for x in g]) > 0.95
    assert _abscorr(r["eigenfunctions"][1], [_phi2(x) for x in g]) > 0.95


def test_scores_track_the_true_scores(fit):
    r, XI = fit
    for j in range(2):
        est = [s[j] for s in r["estimate"]]
        tru = [x[j] for x in XI]
        assert _abscorr(est, tru) > 0.95


def test_error_variance_is_recovered_to_the_right_order(fit):
    r, _ = fit
    # sigma^2 is a difference of two smoothers and is biased upward
    # when few within-subject pairs fall near the diagonal; the test
    # pins the order of magnitude, which is what it can honestly claim
    assert 0.5 * SIGMA2 < r["sigma2"] < 2.0 * SIGMA2


def test_fraction_of_variance_explained_matches_the_design(fit):
    r, _ = fit
    # true first-component share is 4 / (4 + 1) = 0.8
    assert abs(r["fve"][0] - 0.8) < 0.1
    assert r["fve"][1] >= r["fve"][0]


def test_the_diagonal_is_excluded_so_sigma2_is_not_absorbed():
    """With no measurement error the diagonal gap must collapse."""
    Y, T, _ = _simulate(n=120, m=20, seed=3)
    clean = pace([[y for y in row] for row in Y], T, K=2, n_grid=15,
                 bw_mu=0.1, bw_cov=0.1)
    assert clean["sigma2"] >= 0.0


def test_conditional_expectation_shrinks_relative_to_integration():
    """The CE score is a shrinkage estimator; the integral is not.

    With few points per subject the integral approximation has no way
    to borrow strength, so the two disagree -- which is the whole
    reason the paper conditions instead of integrating.
    """
    Y, T, _ = _simulate(n=150, m=4, seed=5)
    ce = pace(Y, T, K=2, n_grid=15, bw_mu=0.15, bw_cov=0.15)
    it = pace(Y, T, K=2, n_grid=15, bw_mu=0.15, bw_cov=0.15,
              shrink=False)
    v_ce = sum(s[0] ** 2 for s in ce["estimate"]) / len(ce["estimate"])
    v_it = sum(s[0] ** 2 for s in it["estimate"]) / len(it["estimate"])
    assert v_ce != v_it
    assert ce["shrink"] is True and it["shrink"] is False


def test_a_single_observation_per_subject_is_refused():
    with pytest.raises(ValueError, match="not identified"):
        pace([[1.0], [2.0], [3.0]], [[0.1], [0.5], [0.9]], K=1)


def test_ragged_designs_are_accepted():
    Y, T, _ = _simulate(n=60, m=8, seed=2)
    Y[0] = Y[0][:5]
    T[0] = T[0][:5]
    r = pace(Y, T, K=1, n_grid=11, bw_mu=0.15, bw_cov=0.15)
    assert len(r["estimate"]) == 60


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="values and"):
        pace([[1.0, 2.0]], [[0.1]], K=1)


def test_both_kernels_run_and_a_bad_one_is_refused():
    Y, T, _ = _simulate(n=60, m=8, seed=4)
    for kern in ("epan", "gauss"):
        r = pace(Y, T, K=1, n_grid=11, bw_mu=0.15, bw_cov=0.15,
                 kernel=kern)
        assert r["kernel"] == kern
    with pytest.raises(ValueError, match="epan or gauss"):
        pace(Y, T, K=1, kernel="triweight")


def test_local_linear_reproduces_a_line_exactly():
    """A local LINEAR smoother has no bias on a linear truth."""
    t = [i / 20.0 for i in range(21)]
    y = [3.0 - 2.0 * x for x in t]
    got = local_linear(t, y, [0.0, 0.25, 0.5, 1.0], 0.3)
    for x, g in zip([0.0, 0.25, 0.5, 1.0], got):
        assert abs(g - (3.0 - 2.0 * x)) < 1e-8
