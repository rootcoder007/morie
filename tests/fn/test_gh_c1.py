"""Tests for the Ghosal Ch 1 Bayes-rule modules."""
from morie.fn.gh_c1_1 import ghosal_bayes_rule_infinite
from morie.fn.gh_c1_2 import ghosal_absolute_continuity
from morie.fn.gh_c1_3 import ghosal_prior_posterior_update
from morie.fn.ghs001 import ghosal_ch1_bayes_formula


def _grid():
    return [i / 100.0 - 3.0 for i in range(601)]


def test_conjugate_posterior_mean_exact():
    # N(0,1) prior, one N(theta,1) obs at 1.0 -> posterior mean 0.5
    r = ghosal_bayes_rule_infinite(_grid())
    assert abs(r["estimate"] - 0.5) < 1e-3


def test_dominated_posterior_matches_and_reports_marginal():
    r = ghosal_absolute_continuity(_grid())
    assert abs(r["estimate"] - 0.5) < 1e-3
    assert r["log_marginal"] < 0


def test_sequential_equals_batch():
    r = ghosal_prior_posterior_update(_grid())
    assert r["sequential_batch_gap"] < 1e-12
    # 3 obs mean 1.0: posterior mean = n*xbar/(n+1) = 0.75
    assert abs(r["estimate"] - 0.75) < 1e-3


def test_set_mass_formula():
    # grid wide enough that truncating the flat prior is negligible
    import math
    supp = [i / 100.0 - 6.0 for i in range(1201)]
    w = [1.0] * len(supp)
    p_theta = lambda t, X: math.exp(-0.5 * (X - t) ** 2)
    r = ghosal_ch1_bayes_formula(lambda t: t > 0.5, 1.0,
                                 p_theta, (supp, w))
    # flat prior, N(1,1) likelihood: P(theta > 0.5 | X) = Phi(0.5)
    from morie.fn._stats_core import norm
    assert abs(r["posterior"] - float(norm.cdf(0.5))) < 5e-3
