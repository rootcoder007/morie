"""Kosorok tranche 2: entropy, GC/Donsker conditions, weak convergence.

The theorems here are implications, so the tests check the SEPARATION
each theorem draws -- classes satisfying the hypothesis vs classes
violating it -- rather than only confirming the easy side."""

import numpy as np
import pytest

from morie.fn.ksr029 import kosorok_ch2_glivenko_cantelli_class
from morie.fn.ksr031 import kosorok_ch2_weak_convergence_tightness
from morie.fn.ksr032 import kosorok_ch2_weak_convergence_iff
from morie.fn.ksr033 import kosorok_ch2_uniform_covering_number
from morie.fn.ksr034 import kosorok_ch2_glivenko_cantelli_bracketing
from morie.fn.ksr035 import kosorok_ch2_donsker_bracketing_integral
from morie.fn.ksr036 import kosorok_ch2_donsker_bracketing_theorem
from morie.fn.ksr037 import kosorok_ch2_glivenko_cantelli_uniform
from morie.fn.ksr038 import kosorok_ch2_donsker_uniform_entropy
from morie.fn.ksr039 import kosorok_ch2_weak_convergence_lipschitz


def _indicators(cuts):
    return [(lambda x, c=c: (np.asarray(x) <= c).astype(float)) for c in cuts]


def test_class_glivenko_cantelli_converges_for_indicators():
    rng = np.random.default_rng(0)
    X = rng.random(3000)
    cuts = np.linspace(0.05, 0.95, 12)
    F = _indicators(cuts)
    out = kosorok_ch2_glivenko_cantelli_class(F, X, P=None)
    assert out["shrinking"] is True
    assert out["n_functions"] == 12
    # with the true P known, the deviation must fall with n
    Ptrue = lambda f: float(np.mean(f(np.linspace(0, 1, 20001))))
    out2 = kosorok_ch2_glivenko_cantelli_class(F, X, P=Ptrue)
    assert out2["sup_deviation"][-1] < out2["sup_deviation"][0]
    with pytest.raises(ValueError):
        kosorok_ch2_glivenko_cantelli_class([], X)


def test_bracketing_numbers_grow_as_eps_shrinks():
    rng = np.random.default_rng(1)
    X = rng.random(200)
    F = _indicators(np.linspace(0.02, 0.98, 40))
    out = kosorok_ch2_glivenko_cantelli_bracketing(F, X)
    assert out["finite_on_grid"] is True
    assert out["monotone"] is True  # counts non-decreasing as eps shrinks
    assert out["bracketing_numbers"][0] <= out["bracketing_numbers"][-1]
    with pytest.raises(ValueError):
        kosorok_ch2_glivenko_cantelli_bracketing(F, X, eps_grid=[0.0])


def test_entropy_integral_separates_polynomial_from_exponential_growth():
    # polynomial bracketing numbers: sqrt(log N) integrable => finite J
    poly = kosorok_ch2_donsker_bracketing_integral(lambda e: (1 / e) ** 3)
    assert poly["finite"] is True
    assert poly["J"] < 5
    # exp(1/eps^2) is exactly the growth rate that makes J DIVERGE:
    # sqrt(log N) = 1/eps, whose integral is logarithmic. The N values
    # overflow to +inf near 0 and that is the mathematically correct
    # outcome, not an error -- capping the exponent would make J
    # finite and destroy the very contrast being tested. errstate
    # silences the notice while keeping the infinity.
    with np.errstate(over="ignore"):
        expo = kosorok_ch2_donsker_bracketing_integral(lambda e: np.exp(1 / e**2))
    assert expo["J"] > poly["J"] * 10
    assert kosorok_ch2_donsker_bracketing_theorem(lambda e: (1 / e) ** 3)[
        "sufficient_condition_met"
    ] is True
    with pytest.raises(ValueError):
        kosorok_ch2_donsker_bracketing_integral(lambda e: 2.0, delta=0.0)


def test_gc_and_donsker_need_different_envelope_moments():
    # finite entropy but a NON-integrable envelope fails GC
    bad_env = kosorok_ch2_glivenko_cantelli_uniform(lambda e: (1 / e) ** 2, np.inf)
    assert bad_env["entropy_finite"] is True
    assert bad_env["envelope_integrable"] is False
    assert bad_env["conditions_met"] is False
    good = kosorok_ch2_glivenko_cantelli_uniform(lambda e: (1 / e) ** 2, 1.5)
    assert good["conditions_met"] is True
    # Donsker keys on the SQUARE of the envelope, not the first moment
    d_ok = kosorok_ch2_donsker_uniform_entropy(lambda e: (1 / e) ** 2, 3.0)
    d_bad = kosorok_ch2_donsker_uniform_entropy(lambda e: (1 / e) ** 2, np.inf)
    assert d_ok["conditions_met"] is True
    assert d_bad["envelope_sq_integrable"] is False
    assert d_bad["conditions_met"] is False


def test_uniform_covering_number_is_scale_free_and_decreasing():
    rng = np.random.default_rng(2)
    X = rng.random(60)
    F = _indicators(np.linspace(0.1, 0.9, 15))
    fine = kosorok_ch2_uniform_covering_number(F, X, eps=0.05, rng=rng)
    coarse = kosorok_ch2_uniform_covering_number(F, X, eps=0.6, rng=rng)
    assert fine["covering_number"] >= coarse["covering_number"]
    assert fine["is_lower_bound"] is True  # sup over Q is sampled
    # scaling every function by 10 leaves the count unchanged: the
    # radius is measured relative to the envelope norm
    F10 = [(lambda x, f=f: 10.0 * f(x)) for f in F]
    scaled = kosorok_ch2_uniform_covering_number(F10, X, eps=0.05,
                                                 rng=np.random.default_rng(2))
    ref = kosorok_ch2_uniform_covering_number(F, X, eps=0.05,
                                              rng=np.random.default_rng(2))
    assert scaled["covering_number"] == ref["covering_number"]
    with pytest.raises(ValueError):
        kosorok_ch2_uniform_covering_number(F, X, eps=1.5)


def test_tightness_separates_a_smooth_process_from_a_shrinking_spike():
    rng = np.random.default_rng(3)
    grid = np.linspace(0, 1, 60)
    # smooth paths: oscillation vanishes with delta => tight
    smooth = np.array([np.sin(2 * np.pi * grid + rng.random() * 6) for _ in range(200)])
    t_smooth = kosorok_ch2_weak_convergence_tightness(smooth, grid, eps=0.3)
    assert t_smooth["decreasing"] is True
    assert t_smooth["probabilities"][-1] < t_smooth["probabilities"][0]
    # a spike of fixed height at a random location keeps a jump at every
    # scale, so the oscillation probability does NOT vanish
    spikes = np.zeros((200, 60))
    for i in range(200):
        spikes[i, rng.integers(1, 59)] = 3.0
    t_spike = kosorok_ch2_weak_convergence_tightness(spikes, grid, eps=0.3)
    assert t_spike["probabilities"][-1] > t_smooth["probabilities"][-1]
    with pytest.raises(ValueError):
        kosorok_ch2_weak_convergence_tightness(smooth, grid, eps=0.0)


def test_weak_convergence_needs_both_halves():
    rng = np.random.default_rng(4)
    grid = np.linspace(0, 1, 40)
    ref = rng.standard_normal((400, 40))
    same = rng.standard_normal((400, 40))
    out = kosorok_ch2_weak_convergence_iff(same, ref, grid, eps=1.5)
    assert out["fidi_converged"] is True
    assert out["weak_convergence"] is True
    # the tolerance is Monte-Carlo-scaled: two samples from the SAME
    # law already differ by ~0.21 in mean at 400 reps over 40 points,
    # so a fixed 0.15 constant would reject identical distributions
    assert out["mean_gap"] > 0.15
    assert out["mean_gap"] < out["mean_tol"]
    # matching marginals but a different scale fails the fidi half
    scaled = rng.standard_normal((400, 40)) * 3.0
    bad = kosorok_ch2_weak_convergence_iff(scaled, ref, grid, eps=1.5)
    assert bad["fidi_converged"] is False
    assert bad["weak_convergence"] is False
    with pytest.raises(ValueError):
        kosorok_ch2_weak_convergence_iff(same, ref[:, :10], grid)


def test_bounded_lipschitz_distance_separates_laws():
    rng = np.random.default_rng(5)
    A = rng.standard_normal(3000)
    B = rng.standard_normal(3000)
    C = rng.standard_normal(3000) + 2.0
    same = kosorok_ch2_weak_convergence_lipschitz(A, B, rng=rng)["bl_distance"]
    diff = kosorok_ch2_weak_convergence_lipschitz(A, C, rng=rng)["bl_distance"]
    assert diff > same * 3  # a shifted law is far in the BL metric
    assert same < 0.15
    with pytest.raises(ValueError):
        kosorok_ch2_weak_convergence_lipschitz([1.0], [2.0])
