"""survvae: deep survival machines (mixture-of-experts survival model).

The generated test imported `vae_survival`, a name the module does not
provide. Rewritten against deep_survival_machines.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.survvae import deep_survival_machines, log_survival


def _cohort(n=48):
    X = [[float(i % 8) / 8.0, 1.0] for i in range(n)]
    times = [1.0 + float(i % 8) for i in range(n)]
    events = [1 if i % 4 else 0 for i in range(n)]
    return X, times, events


def test_fit_returns_the_mixture_it_was_asked_for():
    X, t, e = _cohort()
    r = deep_survival_machines(X, t, e, K=3, restarts=1, seed=0)
    assert r["K"] == 3
    assert len(np.asarray(r["shapes"])) == 3
    assert len(np.asarray(r["scales"])) == 3
    assert all(float(s) > 0 for s in np.asarray(r["scales"]))


def test_the_elbo_lower_bounds_the_exact_loglikelihood():
    """The Jensen gap is non-negative by construction: ELBO <= loglik.
    If this ever inverts, the bound is being computed wrongly."""
    X, t, e = _cohort()
    r = deep_survival_machines(X, t, e, K=2, restarts=1, seed=1)
    assert float(r["elbo"]) <= float(r["loglik"]) + 1e-9
    assert float(r["jensen_gap"]) >= -1e-9


def test_survival_is_monotone_non_increasing_in_time():
    """S(t) is a survival function; it cannot rise."""
    prev = 0.0
    vals = [float(log_survival(t, 1.5, 2.0, "weibull")) for t in (0.5, 1.0, 2.0, 4.0)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))
