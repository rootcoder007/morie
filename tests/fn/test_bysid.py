"""bysid: Bayesian ideal-point estimation.

Armstrong et al., Ch 5 / Ch 6 (Unfolding Analysis of Binary Choice Data,
printed p.129; Bayesian Scaling Models, p.181).
"""

import numpy as np
import pytest

from morie.fn.bysid import bayesian_ideal_points as bip


def _votes(n=120, m=30, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    a = rng.uniform(0.8, 2.0, m)
    b = rng.standard_normal(m)
    p = 1 / (1 + np.exp(-(np.outer(x, a) - b)))
    return x, (rng.random((n, m)) < p).astype(float)


def test_bysid_recovers_the_latent_ordering_but_needs_a_long_chain():
    """This sampler mixes SLOWLY, and a caller needs to know.

    On the same simulated 2PL data where morie.fn.irtsp's EM reaches |r| =
    0.855, the Gibbs sampler here climbs with the chain length:

        n_iter   400 / burn  100   |r| = 0.486
        n_iter  1000 / burn  300   |r| = 0.486
        n_iter  3000 / burn 1000   |r| = 0.618
        n_iter  8000 / burn 3000   |r| = 0.772   <- still rising

    So the default-ish short chains in the generated test were not merely
    imprecise, they were nowhere near the posterior. Anyone using this for
    real ideal points should run tens of thousands of iterations and check
    convergence, or use irtsp if a point estimate is all that is wanted.
    """
    truth, votes = _votes(seed=3)
    xm = np.asarray(bip(votes, n_iter=8000, burn=3000, seed=3)["x_mean"])
    assert abs(float(np.corrcoef(xm, truth)[0, 1])) > 0.7


def test_bysid_recovery_improves_with_chain_length():
    """Guard on the mixing behaviour itself: a longer chain must not be
    worse. If someone 'optimises' the sampler into a fixed point this fails.
    """
    truth, votes = _votes(seed=3)
    short = abs(float(np.corrcoef(
        np.asarray(bip(votes, n_iter=400, burn=100, seed=3)["x_mean"]), truth)[0, 1]))
    long_ = abs(float(np.corrcoef(
        np.asarray(bip(votes, n_iter=8000, burn=3000, seed=3)["x_mean"]), truth)[0, 1]))
    assert long_ > short + 0.1


def test_bysid_credible_interval_brackets_the_posterior_mean():
    """A CI that does not contain its own point estimate is incoherent."""
    _, votes = _votes(n=80, m=20, seed=5)
    r = bip(votes, n_iter=300, burn=80, seed=5)
    xm = np.asarray(r["x_mean"])
    ci = np.asarray(r["x_ci"])
    lo, hi = (ci[:, 0], ci[:, 1]) if ci.ndim == 2 and ci.shape[1] == 2 else (ci[0], ci[1])
    assert np.all(lo <= xm + 1e-9)
    assert np.all(xm <= hi + 1e-9)


def test_bysid_posterior_sd_is_positive():
    _, votes = _votes(n=60, m=15, seed=7)
    sd = np.asarray(bip(votes, n_iter=300, burn=80, seed=7)["x_sd"])
    assert np.all(sd > 0)


def test_bysid_more_votes_shrink_the_posterior_sd():
    """More information must reduce uncertainty -- the basic Bayesian
    guarantee, and the thing a broken sampler gets wrong."""
    sds = []
    for m in (10, 60):
        _, votes = _votes(n=100, m=m, seed=11)
        sds.append(float(np.mean(np.asarray(
            bip(votes, n_iter=300, burn=80, seed=11)["x_sd"]))))
    assert sds[1] < sds[0]


def test_bysid_is_reproducible_for_a_fixed_seed():
    _, votes = _votes(n=50, m=12, seed=13)
    a = np.asarray(bip(votes, n_iter=200, burn=50, seed=99)["x_mean"])
    b = np.asarray(bip(votes, n_iter=200, burn=50, seed=99)["x_mean"])
    assert a == pytest.approx(b, abs=0.0)


def test_bysid_reports_the_iteration_count_and_shapes():
    _, votes = _votes(n=40, m=10, seed=17)
    r = bip(votes, n_iter=250, burn=50, seed=17)
    assert r["n_iter"] == 250
    assert np.asarray(r["x_mean"]).size == 40
    assert np.asarray(r["alpha"]).size == 10
    assert np.asarray(r["beta"]).size == 10
