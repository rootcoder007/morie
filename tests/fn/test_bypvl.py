"""Tests for morie.fn.bypvl: values recomputed from the definition."""

from morie.fn.bypvl import bayesian_p_value


def test_tail_share_of_the_replicated_statistic():
    reps = [0.2, 1.5, 0.9, 2.2, 1.1, 1.0]
    r = bayesian_p_value(reps, 1.0)
    assert r.value == sum(1 for v in reps if v >= 1.0) / len(reps)
    assert abs(r.extra["chain_mean"] - sum(reps) / 6) < 1e-15
