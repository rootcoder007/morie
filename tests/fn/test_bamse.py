"""Tests for morie.fn.bamse: values recomputed from the definition."""

import math

from morie.fn.bamse import bayesian_se_from_posterior


def test_posterior_sd_per_parameter():
    chain = [[1.0, 5.0], [2.0, 3.0], [4.0, 4.0], [0.5, 6.5]]
    r = bayesian_se_from_posterior(chain)
    for k in range(2):
        col = [row[k] for row in chain]
        m = sum(col) / 4
        assert abs(r.extra["ses"][k] - math.sqrt(sum((v - m) ** 2 for v in col) / 3)) < 1e-14
    assert r.value == r.extra["ses"][0]
    assert r.extra["n_samples"] == 4
