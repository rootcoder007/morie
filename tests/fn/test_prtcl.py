"""Tests for prtcl. Full anchor: ledger/wave3/anchor_ts_family.py."""
import math
import pytest
from morie.fn import _array_core as np
from morie.fn import _s03core as k
from morie.fn.prtcl import (effective_sample_size, kalman_filter_1d,
                            particle_filter, systematic_resample)

A_, Q_, C_, R_ = 0.9, 0.4, 1.0, 0.6


@pytest.fixture(scope="module")
def lg():
    rng = np.random.default_rng(5)
    x, ys = 0.0, []
    for _ in range(40):
        x = A_ * x + math.sqrt(Q_) * rng.standard_normal()
        ys.append(C_ * x + math.sqrt(R_) * rng.standard_normal())
    km, kll = kalman_filter_1d(ys, A_, Q_, C_, R_)
    return {"y": ys, "kalman_mean": km, "kalman_ll": kll}


def model():
    def init(g):
        return 0.0

    def step(s, t, g):
        return A_ * s + math.sqrt(Q_) * g.standard_normal()

    def ll(s, o, t):
        return (-0.5 * math.log(2 * math.pi * R_)
                - 0.5 * (o - C_ * s) ** 2 / R_)

    return init, step, ll


def test_systematic_resampling_counts_are_within_one_deterministically():
    w = [0.1, 0.4, 0.3, 0.2]
    idx = systematic_resample(w, u=0.5)
    assert len(idx) == 4
    for j in range(4):
        assert abs(idx.count(j) - 4 * w[j]) < 1.0
    with pytest.raises(ValueError):
        systematic_resample([0.0, 0.0])


def test_effective_sample_size():
    assert effective_sample_size([0.25] * 4) == pytest.approx(4.0)
    assert effective_sample_size([1.0, 0.0, 0.0, 0.0]) == \
        pytest.approx(1.0)


def test_it_reproduces_the_kalman_filter(lg):
    """The closed form is the anchor -- not another particle run."""
    init, step, ll = model()
    pf = particle_filter(lg["y"], 4000, init, step, ll, seed=7)
    gap = k.mean([abs(pf["filtered_mean"][t] - lg["kalman_mean"][t])
                  for t in range(40)])
    assert gap < 0.1
    assert abs(pf["loglik"] - lg["kalman_ll"]) < 1.0
    assert len(pf["ess"]) == 40


def test_the_log_likelihood_is_biased_downward(lg):
    """The LIKELIHOOD is unbiased, so its log is biased down by Jensen
    -- comparing models at different particle counts compares the
    counts."""
    init, step, ll = model()
    biases = []
    for J in (25, 400):
        reps = [particle_filter(lg["y"], J, init, step, ll,
                                seed=100 + r)["loglik"]
                for r in range(12)]
        biases.append(k.mean(reps) - lg["kalman_ll"])
    assert biases[0] < 0.0
    assert abs(biases[1]) < abs(biases[0])


def test_argument_checks(lg):
    init, step, ll = model()
    with pytest.raises(ValueError):
        particle_filter(lg["y"], 1, init, step, ll)
    with pytest.raises(ValueError):
        particle_filter([], 10, init, step, ll)
    with pytest.raises(ValueError):
        particle_filter(lg["y"], 10, init, step, ll,
                        resample_threshold=0.0)
