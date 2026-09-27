"""DCLF and MAD Monte Carlo tests of CSR on Besag's L-function.

The observed statistics equal spatstat.explore dclf.test / mad.test with
Lest and use.theo = TRUE (to 1e-20); tests/cross/test-morie_vs_spatstat.R
repeats that in R.
"""

import math

from morie.fn.csrgt import csr_global_test

PTS = [
    (0.1, 0.2),
    (0.4, 0.8),
    (0.35, 0.3),
    (0.8, 0.6),
    (0.7, 0.15),
    (0.55, 0.5),
    (0.2, 0.65),
    (0.9, 0.9),
    (0.15, 0.95),
    (0.6, 0.85),
]


def test_mad_is_max_deviation_and_documented():
    r = csr_global_test(PTS, (0, 1, 0, 1), nsim=19, statistic="mad")
    assert (round(r.statistic, 6), r.p_value) == (0.206055, 0.05)
    L, rr = r.extra["L"], r.extra["r"]
    assert abs(r.statistic - max(abs(a - b) for a, b in zip(L, rr))) < 1e-15
    assert len(rr) == 513 and abs(rr[-1] - 0.25) < 1e-15


def test_dclf_is_rmax_times_mean_squared_deviation():
    r = csr_global_test(PTS, (0, 1, 0, 1), nsim=3, statistic="dclf")
    L, rr = r.extra["L"], r.extra["r"]
    assert abs(r.statistic - 0.25 * sum((a - b) ** 2 for a, b in zip(L, rr)) / 513) < 1e-15
    assert all(math.isfinite(v) for v in r.extra["simulated"])


def test_simulations_deterministic():
    a = csr_global_test(PTS, (0, 1, 0, 1), nsim=4, statistic="mad")
    b = csr_global_test(PTS, (0, 1, 0, 1), nsim=4, statistic="mad")
    assert a.extra["simulated"] == b.extra["simulated"]
