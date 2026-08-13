"""Tests for pftrep. Full anchor: ledger/wave3/anchor_ts_family.py."""
import math
import pytest
from morie.fn import _array_core as np
from morie.fn.pftrep import logmeanexp, replicated_pfilter
from morie.fn.prtcl import kalman_filter_1d

A_, Q_, C_, R_ = 0.9, 0.4, 1.0, 0.6


@pytest.fixture(scope="module")
def lg():
    rng = np.random.default_rng(5)
    x, ys = 0.0, []
    for _ in range(40):
        x = A_ * x + math.sqrt(Q_) * rng.standard_normal()
        ys.append(C_ * x + math.sqrt(R_) * rng.standard_normal())
    _, kll = kalman_filter_1d(ys, A_, Q_, C_, R_)
    return {"y": ys, "kalman_ll": kll}


def model():
    def init(g):
        return 0.0

    def step(s, t, g):
        return A_ * s + math.sqrt(Q_) * g.standard_normal()

    def ll(s, o, t):
        return (-0.5 * math.log(2 * math.pi * R_)
                - 0.5 * (o - C_ * s) ** 2 / R_)

    return init, step, ll


def test_logmeanexp_averages_likelihoods_not_log_likelihoods():
    assert logmeanexp([0.0, math.log(3.0)]) == pytest.approx(
        math.log(2.0), abs=1e-12)
    # and exceeds the mean of the logs, by Jensen
    assert logmeanexp([0.0, math.log(3.0)]) > (math.log(3.0)) / 2.0
    with pytest.raises(ValueError):
        logmeanexp([])


def test_replication_reports_both_scales_and_the_gap(lg):
    init, step, ll = model()
    r = replicated_pfilter(lg["y"], 200, init, step, ll, n_reps=8,
                           seed=2)
    assert r["jensen_gap"] >= -1e-9
    assert abs(r["loglik"] - lg["kalman_ll"]) < 3.0
    assert len(r["replicates"]) == 8


def test_more_particles_shrink_the_monte_carlo_error(lg):
    init, step, ll = model()
    a = replicated_pfilter(lg["y"], 200, init, step, ll, n_reps=8,
                           seed=2)
    b = replicated_pfilter(lg["y"], 2000, init, step, ll, n_reps=8,
                           seed=2)
    assert b["se"] < a["se"]


def test_argument_checks(lg):
    init, step, ll = model()
    with pytest.raises(ValueError):
        replicated_pfilter(lg["y"], 100, init, step, ll, n_reps=0)
