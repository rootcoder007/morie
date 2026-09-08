"""bayoptr: Bayesian optimisation with EI / UCB acquisitions.

The generated test imported `bayesian_optimization`; the module exports
`bayesian_optimization_ei_ucb`. Rewritten to assert that the optimiser
actually optimises, rather than that its result is a dict.
"""

import pytest

from morie.fn.bayoptr import bayesian_optimization_ei_ucb, resolve_acquisition


def _quadratic(v):
    return (v[0] - 0.3) ** 2


def test_finds_the_minimum_of_a_quadratic():
    """The optimum of (x - 0.3)^2 on [0, 1] is at x = 0.3. A working
    optimiser gets close; a broken one that returns its first random draw
    would not, which is what makes this anchor able to fail.
    """
    r = bayesian_optimization_ei_ucb(_quadratic, [(0.0, 1.0)],
                                     n_iter=25, n_init=5, seed=3)
    assert abs(float(r["x_best"][0]) - 0.3) < 0.1
    assert float(r["y_best"]) < 0.01


def test_reports_every_evaluation_it_made():
    r = bayesian_optimization_ei_ucb(_quadratic, [(0.0, 1.0)],
                                     n_iter=6, n_init=4, seed=1)
    assert r["n_eval"] == 10
    assert len(r["y"]) == 10
    assert float(r["y_best"]) == pytest.approx(min(float(v) for v in r["y"]))


def test_both_acquisitions_are_available():
    # "ucb" is reported back as "lcb": the optimiser minimises, so the
    # upper confidence bound becomes a lower one.
    for asked, reported in (("ei", "ei"), ("ucb", "lcb")):
        r = bayesian_optimization_ei_ucb(_quadratic, [(0.0, 1.0)],
                                         acquisition=asked, n_iter=5,
                                         n_init=3, seed=0)
        assert r["acq"] == reported
    assert resolve_acquisition("ei") is not None
