# SPDX-License-Identifier: AGPL-3.0-or-later
"""Estimate one effect several ways, then correct across the answers.

Parity in contract with rmorie's morie_mrm_estimate_causal_effect. The
anchor is a SIMULATED TRUTH, not this implementation's output, so it can
fail: y = 1 + 0.8 t + 0.5 x, the design in rmorie's own example.
"""

import math
import random

import pytest

import morie


def _sim(n=400, effect=0.8, seed=1, binary_outcome=False):
    random.seed(seed)
    rows = []
    for _ in range(n):
        x = random.gauss(0, 1)
        t = 1 if random.random() < 1 / (1 + math.exp(-0.5 * x)) else 0
        if binary_outcome:
            lp = -0.3 + effect * t + 0.4 * x
            y = 1 if random.random() < 1 / (1 + math.exp(-lp)) else 0
        else:
            y = 1 + effect * t + 0.5 * x + random.gauss(0, 1)
        rows.append({"y": y, "t": t, "x": x})
    return rows


def test_the_estimators_recover_the_simulated_truth():
    eff = morie.mrm_estimate_causal_effect(_sim(), "t", "y", ["x"])
    assert len(eff.results) >= 2
    for r in eff.results:
        assert abs(r["estimate"] - 0.8) < 0.25, r["method"]
    # pooling several noisy answers should not be worse than the worst
    assert abs(eff.consensus["estimate"] - 0.8) < 0.15
    assert eff.spec["n"] == 400


def test_the_consensus_is_inverse_variance_pooled():
    eff = morie.mrm_estimate_causal_effect(_sim(), "t", "y", ["x"])
    w = [1 / r["std_error"] ** 2 for r in eff.results]
    want = sum(wi * r["estimate"] for wi, r in zip(w, eff.results)) / sum(w)
    assert eff.consensus["estimate"] == pytest.approx(want, abs=1e-12)
    assert eff.consensus["std_error"] == pytest.approx((1 / sum(w)) ** 0.5,
                                                       abs=1e-12)


def test_asking_more_than_once_costs_something():
    # THE POINT of the correction: running four estimators and quoting the
    # friendliest is the failure mode, so the adjusted p is never smaller
    # than the raw one.
    eff = morie.mrm_estimate_causal_effect(_sim(), "t", "y", ["x"])
    for r in eff.results:
        assert r["p_adjusted"] >= r["p_value"] - 1e-15
    raw = morie.mrm_estimate_causal_effect(_sim(), "t", "y", ["x"],
                                           correction="none")
    for r in raw.results:
        assert r["p_adjusted"] == pytest.approx(r["p_value"])


def test_confidence_intervals_bracket_the_estimate():
    eff = morie.mrm_estimate_causal_effect(_sim(), "t", "y", ["x"])
    for r in eff.results:
        assert r["ci_lower"] < r["estimate"] < r["ci_upper"]
        half = 1.959963984540054 * r["std_error"]
        assert r["ci_upper"] - r["estimate"] == pytest.approx(half, abs=1e-12)


def test_nothing_requested_is_silently_dropped():
    # A consensus over an unknown subset is not a consensus, so every
    # requested method must land in exactly one of results or failed.
    methods = ("matching", "ate", "aipw", "dml")
    eff = morie.mrm_estimate_causal_effect(_sim(), "t", "y", ["x"],
                                           methods=methods)
    assert len(eff.results) + len(eff.failed) == len(methods)
    assert not (set(eff.failed) & {r["method"] for r in eff.results})


def test_aipw_runs_on_a_continuous_outcome():
    # The outcome model has to match the outcome: leaving the logistic
    # default on a continuous outcome fails inside the native core with
    # "binary only", which reads like a missing capability and is really
    # the wrong link function.
    eff = morie.mrm_estimate_causal_effect(_sim(), "t", "y", ["x"],
                                           methods=("aipw",))
    assert eff.failed == {}
    assert abs(eff.results[0]["estimate"] - 0.8) < 0.2


def test_aipw_runs_on_a_binary_outcome():
    eff = morie.mrm_estimate_causal_effect(
        _sim(n=600, effect=0.9, seed=7, binary_outcome=True),
        "t", "y", ["x"], methods=("aipw", "ate"))
    assert eff.failed == {}
    got = {r["method"].split(" (")[0]: r["estimate"] for r in eff.results}
    # both are risk differences here, so they should broadly agree
    assert abs(got["aipw"] - got["ipw ate"]) < 0.05


def test_a_categorical_treatment_is_refused():
    rows = [{"y": 1.0, "t": "high", "x": 0.1},
            {"y": 2.0, "t": "low", "x": 0.2}]
    with pytest.raises(ValueError, match="binary 0/1 treatment"):
        morie.mrm_estimate_causal_effect(rows, "t", "y", ["x"])


def test_inputs_are_checked_before_any_estimator_runs():
    rows = _sim(n=20)
    with pytest.raises(ValueError, match="missing column"):
        morie.mrm_estimate_causal_effect(rows, "t", "y", ["nope"])
    with pytest.raises(ValueError, match="unknown method"):
        morie.mrm_estimate_causal_effect(rows, "t", "y", ["x"],
                                         methods=("bogus",))
    with pytest.raises(ValueError, match="unknown correction"):
        morie.mrm_estimate_causal_effect(rows, "t", "y", ["x"],
                                         methods=("dml",),
                                         correction="bogus")
    with pytest.raises(ValueError, match="must not be empty"):
        morie.mrm_estimate_causal_effect([], "t", "y", ["x"])


def test_every_estimator_failing_is_an_error_not_an_empty_answer():
    # Fewer rows than cross-fitting folds: DML cannot run, and with it
    # the only requested method, so there is no answer to report. An
    # empty results table would look like a finding of nothing.
    rows = _sim(n=4)
    with pytest.raises(RuntimeError, match="every requested estimator"):
        morie.mrm_estimate_causal_effect(rows, "t", "y", ["x"],
                                         methods=("dml",))


def test_the_effect_renders_through_the_report():
    eff = morie.mrm_estimate_causal_effect(_sim(), "t", "y", ["x"],
                                           methods=("ate", "dml"))
    out = morie.mrm_report(effect=eff)
    assert "Causal effect" in out and "consensus" in out
    assert "dml plr" in out
