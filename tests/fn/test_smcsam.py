"""Tests for smcsam and smcopt (Del Moral, Doucet & Jasra 2006)."""

import math

from morie.fn import _array_core as np

from morie.fn.smcopt import annealing_ladder, smcopt
from morie.fn.smcsam import (ess, random_walk_kernel, resample, smcsam,
                             smc_sampler, temperature_ladder)

PRIOR_V, LIK_V, MU = 25.0, 0.25, 2.0


def _log_gamma(x, phi):
    return (-0.5 * x[0] ** 2 / PRIOR_V +
            phi * (-0.5 * (x[0] - MU) ** 2 / LIK_V))


def test_ess_identities():
    assert abs(ess([1.0] * 10) - 10.0) < 1e-12
    assert abs(ess([1.0] + [0.0] * 9) - 1.0) < 1e-12
    assert abs(ess([0.5, 0.25, 0.25]) - 1.0 / 0.375) < 1e-12


def test_resampling_is_unbiased():
    rng = np.random.default_rng(5)
    w = [0.5, 0.3, 0.15, 0.05]
    for scheme in ("multinomial", "stratified", "systematic", "residual"):
        counts = [0.0] * 4
        reps = 1500
        for _ in range(reps):
            idx = resample(w, rng, scheme)
            assert len(idx) == 4
            for i in idx:
                counts[i] += 1
        got = [c / (reps * 4) for c in counts]
        assert all(abs(got[j] - w[j]) < 0.03 for j in range(4))


def test_conjugate_target_is_recovered():
    a, b = 1.0 / PRIOR_V, 1.0 / LIK_V
    prec = a + b
    post_mean, post_var = b * MU / prec, 1.0 / prec
    want_log_ratio = (0.5 * math.log(2 * math.pi / prec) -
                      0.5 * (a * b / prec) * MU ** 2 -
                      0.5 * math.log(2 * math.pi * PRIOR_V))
    r = smcsam(_log_gamma,
               lambda g: [math.sqrt(PRIOR_V) * g.standard_normal()],
               n_particles=1200, n_steps=30,
               kernel=random_walk_kernel(scale=1.0, n_moves=3), seed=1)
    assert abs(r["mean"][0] - post_mean) < 0.08
    assert abs(r["variance"][0] - post_var) < 0.08
    assert abs(r["log_norm_const"] - want_log_ratio) < 0.15
    assert abs(sum(r["weights"]) - 1.0) < 1e-12


def test_equation_31_with_a_frozen_kernel():
    def no_move(x, log_target, rng):
        return list(x), 0.0

    r = smcsam(_log_gamma, lambda g: [0.7], n_particles=4, n_steps=3,
               kernel=no_move, ess_threshold=1e-9, seed=2)
    phis = r["ladder"]
    want = sum(_log_gamma([0.7], phis[n]) - _log_gamma([0.7], phis[n - 1])
               for n in range(1, len(phis)))
    assert abs(r["log_norm_const"] - want) < 1e-9


def test_one_step_tempering_degenerates():
    sharp = smcsam(_log_gamma,
                   lambda g: [math.sqrt(PRIOR_V) * g.standard_normal()],
                   n_particles=400, ladder=[0.0, 1.0],
                   kernel=random_walk_kernel(scale=1.0),
                   ess_threshold=1e-9, seed=3)
    gradual = smcsam(_log_gamma,
                     lambda g: [math.sqrt(PRIOR_V) * g.standard_normal()],
                     n_particles=400, n_steps=30,
                     kernel=random_walk_kernel(scale=1.0),
                     ess_threshold=1e-9, seed=3)
    assert sharp["ess"] < gradual["ess"]


def test_ladders():
    lad = temperature_ladder(5)
    assert lad[0] == 0.0 and lad[-1] == 1.0
    assert all(lad[i] < lad[i + 1] for i in range(4))
    ann = annealing_ladder(6, 50.0, 0.1)
    assert abs(ann[0] - 0.1) < 1e-12 and abs(ann[-1] - 50.0) < 1e-9


def test_optimisation_finds_the_narrow_global_maximum():
    def f(x):
        return max(3.0 * math.exp(-20.0 * (x[0] - 2.0) ** 2),
                   2.0 * math.exp(-2.0 * x[0] ** 2))

    o = smcopt(f, lambda g: [6.0 * g.random() - 3.0], n_particles=250,
               n_steps=40, phi_max=80.0, seed=2)
    assert abs(o["best_x"][0] - 2.0) < 0.05
    assert abs(o["best_value"] - f(o["best_x"])) < 1e-12
    mn = smcopt(lambda x: (x[0] - 1.5) ** 2,
                lambda g: [8.0 * g.random() - 4.0], n_particles=200,
                n_steps=30, phi_max=200.0, seed=4, maximise=False)
    assert abs(mn["best_x"][0] - 1.5) < 0.05


def test_validation():
    rng = np.random.default_rng(1)
    for call in (lambda: smcsam(_log_gamma, lambda g: [0.0], ladder=[0.5]),
                 lambda: smcsam(_log_gamma, lambda g: [0.0], n_particles=1),
                 lambda: smcsam(_log_gamma, lambda g: [0.0],
                                ess_threshold=0.0),
                 lambda: smcsam(_log_gamma, lambda g: [0.0],
                                weight_rule="general"),
                 lambda: resample([1.0, 1.0], rng, "best"),
                 lambda: ess([0.0, 0.0]),
                 lambda: temperature_ladder(5, "linear-ish"),
                 lambda: annealing_ladder(5, 1.0, 2.0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert smc_sampler is smcsam
