"""Tests for airdisp: atmospheric dispersion models."""

import math

from morie.fn._rng import random_normal
from morie.fn.airdisp import (
    advection_diffusion_2d,
    briggs_plume_rise,
    gaussian_plume,
    gaussian_puff,
    lagrangian_particles,
    pg_sigmas,
)


def test_pg_sigmas_briggs_rural_and_urban():
    x = [100.0, 2500.0]
    r = pg_sigmas(x, "C")
    for i, v in enumerate(x):
        assert abs(r.sigma_y[i] - 0.11 * v / math.sqrt(1 + 1e-4 * v)) <= 1e-12 * v
        assert abs(r.sigma_z[i] - 0.08 * v / math.sqrt(1 + 2e-4 * v)) <= 1e-12 * v
    u = pg_sigmas(x, "A", setting="urban")
    for i, v in enumerate(x):
        assert abs(u.sigma_z[i] - 0.24 * v * math.sqrt(1 + 1e-3 * v)) <= 1e-12 * v
    f = pg_sigmas(x, "F")
    assert abs(f.sigma_z[1] - 0.016 * 2500 / (1 + 0.0003 * 2500)) <= 1e-12


def test_briggs_rise_branches():
    F = 9.80616 * 12 * 1.0 * (420 - 285) / (4 * 420)
    r = briggs_plume_rise([10.0, 1e5], 4.0, diameter=1.0, exit_velocity=12.0, stack_temp=420.0, ambient_temp=285.0)
    assert abs(r.flux - F) <= 1e-12 and F < 55
    assert abs(r.final_rise - 21.425 * F**0.75 / 4) <= 1e-12
    assert abs(r.x_final - 49 * F ** (5 / 8)) <= 1e-12
    assert abs(r.rise[0] - 1.6 * F ** (1 / 3) * 10 ** (2 / 3) / 4) <= 1e-12
    big = briggs_plume_rise([1e5], 4.0, diameter=4.0, exit_velocity=12.0, stack_temp=420.0, ambient_temp=285.0)
    assert big.flux >= 55 and abs(big.rise[0] - 38.71 * big.flux**0.6 / 4) <= 1e-12
    st = briggs_plume_rise(
        [1e5], 4.0, diameter=1.0, exit_velocity=12.0, stack_temp=420.0, ambient_temp=285.0, stability="E"
    )
    s = 9.80616 * 0.02 / 285
    assert abs(st.final_rise - 2.6 * (F / (4 * s)) ** (1 / 3)) <= 1e-12


def test_plume_ground_release_centreline():
    s = pg_sigmas([800.0], "D")
    c = gaussian_plume(10.0, 3.0, 0.0, [(800.0, 0.0, 0.0), (-5.0, 0.0, 0.0)])
    assert abs(c[0] - 10.0 / (math.pi * 3.0 * s.sigma_y[0] * s.sigma_z[0])) <= 1e-15
    assert c[1] == 0.0
    # a very high lid adds nothing; a lid doubles the mass trapped near the ground far downwind
    far = gaussian_plume(10.0, 3.0, 50.0, [(800.0, 20.0, 5.0)], mixing_height=1e6)
    assert abs(far[0] - gaussian_plume(10.0, 3.0, 50.0, [(800.0, 20.0, 5.0)])[0]) <= 1e-18


def test_puff_formula():
    s = pg_sigmas([600.0], "B", setting="urban")
    sy, sz = s.sigma_y[0], s.sigma_z[0]
    c = gaussian_puff(5.0, 2.0, 10.0, [(550.0, 7.0, 3.0)], 300.0, stability="B", setting="urban")[0]
    v = math.exp(-(7.0**2) / (2 * sz * sz)) + math.exp(-(13.0**2) / (2 * sz * sz))
    want = (
        5.0
        / ((2 * math.pi) ** 1.5 * sy * sy * sz)
        * math.exp(-(50.0**2) / (2 * sy * sy))
        * math.exp(-49 / (2 * sy * sy))
        * v
    )
    assert abs(c - want) <= 1e-12 * want


def test_advection_diffusion_exact_shift_and_diffusion_stencil():
    c = [[0.0] * 6 for _ in range(6)]
    c[2][3] = 1.0
    r = advection_diffusion_2d(c, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1)
    assert r.field[3][3] == 1.0 and r.field[2][3] == 0.0 and r.cfl == 1.0
    d = advection_diffusion_2d(c, 0.0, 0.0, 0.1, 0.2, 1.0, 1.0, 1.0, 1)
    assert abs(d.field[2][3] - (1 - 2 * 0.1 - 2 * 0.2)) <= 1e-15
    assert abs(d.field[1][3] - 0.1) <= 1e-15 and abs(d.field[2][4] - 0.2) <= 1e-15
    assert abs(d.mass - 1.0) <= 1e-15


def test_lagrangian_random_walk_reuses_philox_streams():
    r = lagrangian_particles(7, 1.0, 2.0, 0.8, 0.1, 0.5, 0.3, 0.7, 3, seed=5)
    x = [1.0] * 7
    for k in range(3):
        e = random_normal(7, seed=5, stream=2 * k)
        x = [x[i] + 0.8 * 0.7 + math.sqrt(2 * 0.5 * 0.7) * float(e[i]) for i in range(7)]
    assert max(abs(a - b) for a, b in zip(r.x, x)) <= 1e-12
    g = lagrangian_particles(40, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2, grid=(-10, 10, -10, 10, 4, 4))
    assert abs(sum(sum(row) for row in g.concentration) * 25 - 1.0) <= 1e-12
