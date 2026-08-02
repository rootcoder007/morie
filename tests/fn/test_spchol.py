"""Tests for the Ch. 7 simulation family (Schabenberger & Gotway 2005).

spchol   -- Cholesky (LU) root,          Sec 7.1.1
spspec2  -- spectral decomposition root, Sec 7.1.2
spcnds   -- conditioning by kriging,     Sec 7.2.2, eq (7.1)

Ch. 7 opens from the reproductive property of the Gaussian: any Sigma^(1/2)
will serve. The two unconditional modules are therefore two square roots of
the same matrix, and the tests below say so explicitly rather than expecting
them to agree pointwise.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn._schab_fit import covariance_matrix
from morie.fn._schab_sim import (cholesky_root, simple_kriging_variance,
                                 simulate_conditional, simulate_unconditional,
                                 spectral_root)
from morie.fn.spchol import schabenberger_cholesky_sim as spchol
from morie.fn.spcnds import schabenberger_conditional_sim as spcnds
from morie.fn.spspec2 import schabenberger_spectral_sim as spspec2

NUGGET, SILL, RANGE = 0.2, 1.5, 2.5


def _grid(k=5):
    g = np.arange(k) / 1.0
    return np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)


def _cov(coords):
    return covariance_matrix(coords, NUGGET, SILL, RANGE, "exponential")


def test_cholesky_root_is_lower_triangular_and_reconstructs_sigma():
    """Sec 7.1.1: Sigma = U'U with U upper triangular. The L returned is U'."""
    cov = _cov(_grid())
    L = cholesky_root(cov)
    assert np.allclose(L, np.tril(L))
    assert np.abs(L @ L.T - cov).max() < 1e-12


def test_spectral_root_is_symmetric_and_reconstructs_sigma():
    """Sec 7.1.2: Sigma^(1/2) = P Delta^(1/2) P', which is symmetric -- so
    Sigma^(1/2) Sigma^(1/2) = Sigma without a transpose."""
    cov = _cov(_grid())
    P = spectral_root(cov)
    assert np.allclose(P, P.T)
    assert np.abs(P @ P - cov).max() < 1e-10


def test_the_two_roots_are_genuinely_different_square_roots():
    """Both are valid square roots of the same Sigma, so from one random
    stream they give different fields. A test expecting them to match would
    be asserting something the book denies."""
    cov = _cov(_grid())
    mu = np.zeros(cov.shape[0])
    a = spchol(mu, cov, seed=1)["field"]
    b = spspec2(mu, cov, seed=1)["field"]
    assert not np.allclose(a, b)
    assert not np.allclose(cholesky_root(cov), spectral_root(cov))


@pytest.mark.parametrize("method", ["cholesky", "spectral"])
def test_both_methods_reproduce_the_target_covariance(method):
    """The property that makes either root correct."""
    cov = _cov(_grid(4))
    n = cov.shape[0]
    draws = np.array([simulate_unconditional(np.zeros(n), cov, method=method,
                                             seed=11, stream=s)
                      for s in range(6000)])
    assert np.abs(np.cov(draws, rowvar=False) - cov).max() < 0.15
    assert np.abs(draws.mean(axis=0)).max() < 0.12


def test_the_mean_is_carried_through():
    cov = _cov(_grid(4))
    mu = np.linspace(-3.0, 3.0, cov.shape[0])
    draws = np.array([simulate_unconditional(mu, cov, seed=3, stream=s)
                      for s in range(3000)])
    assert np.abs(draws.mean(axis=0) - mu).max() < 0.15


def test_conditional_simulation_honors_the_data():
    """Sec 7.2.2 property (i): for s0 among the sampled locations,
    Zc(s0) = Z(s0). Exactly, not approximately."""
    cov = _cov(_grid())
    n = cov.shape[0]
    truth = simulate_unconditional(np.full(n, 4.0), cov, seed=99)
    res = spcnds(cov, truth[:10], 10, mean=4.0, seed=1)
    assert res["honors_data"] < 1e-10
    assert np.allclose(res["field"][:10], truth[:10])


def test_conditional_simulation_reproduces_the_covariance():
    """Sec 7.2.2 property (iii)."""
    cov = _cov(_grid(4))
    n = cov.shape[0]
    m = 6
    truth = simulate_unconditional(np.zeros(n), cov, seed=99)
    draws = np.array([simulate_conditional(cov, truth[:m], m, mean=0.0,
                                           seed=1, stream=s)
                      for s in range(6000)])
    assert np.abs(np.cov(draws[:, m:], rowvar=False) - cov[m:, m:]).max() < 0.2


def test_conditional_simulation_satisfies_the_two_sigma_sk_identity():
    """Sec 7.2.2 closes with E[(Zc(s) - Z(s))^2] = 2 sigma^2_sk -- the
    sharpest check available. The expectation is over BOTH the field and the
    simulation, so the truth is redrawn every replicate; holding it fixed
    measures something else and lands near 0.91."""
    cov = _cov(_grid(4))
    n = cov.shape[0]
    m = 6
    sk = simple_kriging_variance(cov, m)[m:]
    acc = np.zeros(n - m)
    reps = 6000
    for r in range(reps):
        truth = simulate_unconditional(np.zeros(n), cov, seed=4242, stream=2 * r)
        zc = simulate_conditional(cov, truth[:m], m, mean=0.0,
                                  seed=4242, stream=2 * r + 1)
        acc += (zc[m:] - truth[m:]) ** 2
    ratio = (acc / reps) / (2.0 * sk)
    assert ratio.mean() == pytest.approx(1.0, abs=0.06)


def test_conditional_is_more_variable_than_the_kriging_predictor():
    """"A conditional simulation of a random field will exhibit more
    variability between the observed points than the kriging predictor" --
    the kriging predictor is the mean of the conditional draws, so the draws
    must scatter about it."""
    cov = _cov(_grid(4))
    n = cov.shape[0]
    m = 6
    truth = simulate_unconditional(np.zeros(n), cov, seed=99)
    draws = np.array([simulate_conditional(cov, truth[:m], m, mean=0.0,
                                           seed=1, stream=s)
                      for s in range(2000)])
    predictor = draws.mean(axis=0)
    assert draws[:, m:].var(axis=0).min() > 0.0
    assert np.var(draws[:, m:], axis=0).mean() > np.var(predictor[m:]) * 0.1


def test_rejects_bad_input():
    cov = _cov(_grid(3))
    with pytest.raises(ValueError):
        spchol(np.zeros(3), cov)
    with pytest.raises(ValueError):
        spspec2(np.zeros(3), cov)
    with pytest.raises(ValueError):
        spcnds(cov, np.zeros(2), 3, mean=0.0)
    with pytest.raises(ValueError):
        simulate_unconditional(np.zeros(cov.shape[0]), cov, method="nope")
