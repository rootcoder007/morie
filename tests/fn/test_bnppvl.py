"""bnppvl: Beta quantile pyramid, Bayesian nonparametric predictive value.

The generated test imported `bnp_predictive_value`; the entry point is
`bnppvl`. Rewritten against it, anchored on the predictive distribution
being a distribution.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.bnppvl import bnppvl

X = [0.12, 0.24, 0.31, 0.44, 0.52, 0.61, 0.73, 0.85]


def test_predictive_quantiles_are_ordered_and_in_range():
    """Quantiles for increasing probabilities cannot decrease, and the
    support here is [0, 1]."""
    r = bnppvl(X, sweeps=120, burn=40, seed=0)
    q = [float(v) for v in np.asarray(r["predictive_quantile"])]
    assert q == sorted(q)
    assert all(0.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in q)


def test_the_cdf_is_monotone_where_it_is_evaluated():
    """`cdf` is the predictive CDF at the observed points, not over the
    whole support, so it need not reach 1 -- but it cannot decrease."""
    r = bnppvl(X, sweeps=120, burn=40, seed=0)
    cdf = [float(v) for v in np.asarray(r["cdf"])]
    assert len(cdf) == len(X)
    assert all(b >= a - 1e-9 for a, b in zip(cdf, cdf[1:]))
    assert all(0.0 <= v <= 1.0 for v in cdf)


def test_the_predictive_mean_sits_inside_the_data_range():
    r = bnppvl(X, sweeps=120, burn=40, seed=0)
    assert min(X) - 0.25 < float(r["predictive_mean"]) < max(X) + 0.25
    assert float(r["predictive_sd"]) > 0
