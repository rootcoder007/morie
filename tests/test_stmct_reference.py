"""Diggle's space-time interaction Monte Carlo test.

The statistic equals splancs::stmctest's t0 to 1e-10 (splancs' pi constant);
tests/cross/test-morie_vs_splancs.R repeats that in R.
"""

from morie.fn.stk import space_time_k
from morie.fn.stmct import space_time_interaction_test

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
TM = [1.0, 7.0, 2.0, 5.0, 3.0, 4.5, 6.0, 9.0, 8.0, 8.5]


def test_statistic_is_sum_of_d_and_documented():
    r = space_time_interaction_test(PTS, TM, [0.2, 0.4], [2.3, 4.3], window=(0, 1, 0, 1), tlimits=(0, 10), nsim=19)
    k = space_time_k(PTS, TM, [0.2, 0.4], [2.3, 4.3], window=(0, 1, 0, 1), tlimits=(0, 10))
    assert abs(r.statistic - sum(float(v) for row in k.extra["D"].tolist() for v in row)) < 1e-12
    assert (round(r.statistic, 6), r.p_value) == (3.960941, 0.05)
    assert len(r.extra["simulated"]) == 19


def test_permutation_is_deterministic_and_seed_dependent():
    a = space_time_interaction_test(PTS, TM, [0.2, 0.4], [2.3, 4.3], window=(0, 1, 0, 1), tlimits=(0, 10), nsim=5)
    b = space_time_interaction_test(PTS, TM, [0.2, 0.4], [2.3, 4.3], window=(0, 1, 0, 1), tlimits=(0, 10), nsim=5)
    c = space_time_interaction_test(
        PTS, TM, [0.2, 0.4], [2.3, 4.3], window=(0, 1, 0, 1), tlimits=(0, 10), nsim=5, seed=2
    )
    assert a.extra["simulated"] == b.extra["simulated"]
    assert a.extra["simulated"] != c.extra["simulated"]
