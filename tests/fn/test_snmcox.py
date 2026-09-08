"""Tests for morie.fn.snmcox -- g-estimation of a structural nested
failure time model (Robins 1992).

The strongest anchor here is parameter recovery: data are generated from
a KNOWN psi and the estimator has to find it. The second is that the
method beats the naive analysis on a confounded design, which is the
whole reason structural nested models exist.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.snmcox import snmcox, blip_down, gest_score


def _design(n, psi_true, seed=12345, confounded=False):
    rng = np.random.default_rng(seed)
    U, L, A = [], [], []
    for _ in range(n):
        l = 1.0 if rng.random() < 0.5 else 0.0
        base = -math.log(1.0 - rng.random()) * 5.0 + 1.0
        U.append(base * (0.5 if (confounded and l) else 1.0))
        L.append([l])
        A.append(1.0 if rng.random() < (0.2 + 0.6 * l) else 0.0)
    T = [U[i] * math.exp(-psi_true * A[i]) for i in range(n)]
    return T, [1.0] * n, A, L, U


def test_blip_down_never_treated_is_the_identity():
    """U(psi) = T for a subject never on treatment, at every psi."""
    for psi in (-1.0, 0.0, 0.7, 3.0):
        assert blip_down(7.0, [], psi) == pytest.approx(7.0, abs=1e-15)


def test_blip_down_always_treated_is_T_exp_psi():
    for psi in (-0.5, 0.0, 0.9):
        assert blip_down(7.0, [(0.0, 7.0)], psi) == pytest.approx(
            7.0 * math.exp(psi), rel=1e-14)


def test_blip_down_partial_treatment_splits_the_integral():
    assert blip_down(10.0, [(0.0, 5.0)], 0.5) == pytest.approx(
        5.0 + 5.0 * math.exp(0.5), rel=1e-14)


def test_blip_down_clips_treatment_beyond_the_failure_time():
    """Treatment recorded past the event must not contribute."""
    assert blip_down(4.0, [(0.0, 99.0)], 0.3) == pytest.approx(
        4.0 * math.exp(0.3), rel=1e-14)


def test_psi_zero_leaves_every_time_unchanged():
    for hist in ([], [(0.0, 3.0)], [(1.0, 2.0), (4.0, 6.0)]):
        assert blip_down(7.0, hist, 0.0) == pytest.approx(7.0, abs=1e-15)


def test_recovers_a_known_psi():
    """Data built from psi = 0.6; the estimator must find it."""
    T, ev, A, L, _ = _design(4000, 0.6)
    r = snmcox(T, ev, A, L)
    assert r["estimate"] == pytest.approx(0.6, abs=0.08)
    assert r["lower"] <= 0.6 <= r["upper"]
    assert abs(r["score_at_estimate"]) < 1e-4
    assert r["converged"] is True


def test_null_effect_gives_psi_near_zero():
    T, ev, A, L, _ = _design(4000, 0.0)
    r = snmcox(T, ev, A, L)
    assert abs(r["estimate"]) < 0.08
    assert r["lower"] <= 0.0 <= r["upper"]


def test_estimator_is_equivariant_in_psi():
    """Generating at psi0 + delta instead of psi0 must shift the estimate
    by exactly delta: the blip-down is an exact group action, so any
    sampling error is common to both fits."""
    T0, ev, A, L, _ = _design(1000, 0.0)
    T1, _, _, _, _ = _design(1000, 0.6)
    r0 = snmcox(T0, ev, A, L)
    r1 = snmcox(T1, ev, A, L)
    assert r1["estimate"] - r0["estimate"] == pytest.approx(0.6, abs=1e-6)


def test_beats_the_naive_analysis_under_confounding():
    """The confounder shortens survival and raises treatment probability,
    so comparing treated with untreated gets the sign wrong. G-estimation
    must not."""
    T, ev, A, L, _ = _design(4000, 0.5, seed=999, confounded=True)
    treated = [T[i] for i in range(len(T)) if A[i] > 0]
    untreated = [T[i] for i in range(len(T)) if A[i] == 0]
    naive = (math.log(sum(treated) / len(treated))
             - math.log(sum(untreated) / len(untreated)))
    r = snmcox(T, ev, A, L)
    assert naive < 0.0                     # naive says treatment HARMS
    assert abs(r["estimate"] - 0.5) < abs(naive - 0.5)
    assert abs(r["estimate"] - 0.5) < 0.1


def test_time_ratio_is_exp_psi():
    T, ev, A, L, _ = _design(500, 0.4)
    r = snmcox(T, ev, A, L)
    assert r["time_ratio"] == pytest.approx(math.exp(r["estimate"]), rel=1e-12)


def test_blipping_at_the_true_psi_restores_the_latent_times():
    T, ev, A, L, U = _design(200, 0.6)
    for i in range(len(T)):
        hist = [(0.0, T[i])] if A[i] > 0 else []
        assert blip_down(T[i], hist, 0.6) == pytest.approx(U[i], rel=1e-12)


def test_score_crosses_zero_exactly_once_over_the_grid():
    T, ev, A, L, _ = _design(1000, 0.3)
    r = snmcox(T, ev, A, L)
    s = r["grid_score"]
    crossings = sum(1 for i in range(len(s) - 1) if s[i] * s[i + 1] < 0.0)
    assert crossings == 1


def test_censoring_refuses_rather_than_guessing():
    """The artificial-censoring construction is not implemented. Two
    approximations were tried and both failed the recovery anchor -- one
    recovered -0.05 for a true 0.5 -- so the path raises instead of
    returning a number that would look plausible."""
    T, ev, A, L, _ = _design(100, 0.5)
    with pytest.raises(NotImplementedError, match="artificial-censoring"):
        snmcox(T, ev, A, L, censor_time=[v * 1.3 for v in T])


def test_inputs_are_validated():
    with pytest.raises(ValueError, match="event"):
        snmcox([1.0, 2.0], [0.0, 2.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="treatment values"):
        snmcox([1.0, 2.0], [1.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="psi_range"):
        snmcox([1.0, 2.0], [1.0, 1.0], [1.0, 0.0], psi_range=(2.0, 1.0))
    with pytest.raises(ValueError, match="negative"):
        blip_down(-1.0, [], 0.5)
