"""Tests for distq -- the C51 categorical projection.

Replaces a generated test that called a stub returning mean(y). Full
anchor: ledger/wave3/anchor_distq.py.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.distq import (atoms, bernoulli_algorithm, c51_update,
                            categorical_loss, categorical_projection,
                            distribution_mean, greedy_action,
                            value_distribution_iteration)

VMIN, VMAX, NA = -10.0, 10.0, 51


@pytest.fixture(scope="module")
def grid():
    z, dz = atoms(VMIN, VMAX, NA)
    rng = np.random.default_rng(3)
    raw = [float(rng.uniform()) + 0.01 for _ in range(NA)]
    return {"z": z, "dz": dz, "p": [v / sum(raw) for v in raw]}


def test_the_support_is_the_papers(grid):
    assert len(grid["z"]) == 51
    assert grid["z"][0] == -10.0 and grid["z"][-1] == 10.0
    assert grid["dz"] == pytest.approx(0.4)


@pytest.mark.parametrize("r,g", [
    (0.37, 0.99), (0.0, 1.0), (0.0, 0.0), (100.0, 0.9),
    (-100.0, 0.9), (0.4, 1.0), (0.8, 0.5), (2.0, 0.0)])
def test_mass_is_preserved(grid, r, g):
    """Including the exact-hit case: when b lands on an atom, l == u and
    Algorithm 1 written literally adds p*(u-b) + p*(b-l) = 0, losing the
    mass entirely."""
    m = categorical_projection(r, g, grid["p"], VMIN, VMAX)
    assert sum(m) == pytest.approx(1.0, abs=1e-12)
    assert all(v >= 0.0 for v in m)


def test_gamma_zero_on_an_atom_keeps_all_mass_there(grid):
    m = categorical_projection(2.0, 0.0, grid["p"], VMIN, VMAX)
    carrying = [i for i in range(NA) if m[i] > 0.0]
    assert carrying == [grid["z"].index(2.0)]
    assert m[carrying[0]] == pytest.approx(1.0, abs=1e-12)
    # and between atoms it splits over exactly two
    m2 = categorical_projection(2.2, 0.0, grid["p"], VMIN, VMAX)
    assert sum(1 for v in m2 if v > 0.0) == 2


def test_r_zero_gamma_one_is_the_identity(grid):
    m = categorical_projection(0.0, 1.0, grid["p"], VMIN, VMAX)
    for i in range(NA):
        assert m[i] == pytest.approx(grid["p"][i], abs=1e-14)


def test_done_equals_gamma_zero(grid):
    a = categorical_projection(1.3, 0.97, grid["p"], VMIN, VMAX,
                               done=True)
    b = categorical_projection(1.3, 0.0, grid["p"], VMIN, VMAX)
    assert a == pytest.approx(b, abs=1e-15)


@pytest.mark.parametrize("r,g", [(0.37, 0.9), (-1.1, 0.5), (2.0, 0.25)])
def test_the_projected_mean_is_r_plus_gamma_ez(grid, r, g):
    """A closed form the projection has to reproduce exactly."""
    narrow = [0.0] * NA
    for i in range(20, 31):
        narrow[i] = 1.0 / 11.0
    m = categorical_projection(r, g, narrow, VMIN, VMAX)
    want = r + g * distribution_mean(narrow, grid["z"])
    assert distribution_mean(m, grid["z"]) == pytest.approx(want,
                                                            abs=1e-12)


def test_clipping_moves_the_mean_and_piles_on_the_boundary(grid):
    """So the closed-form check above is not vacuous."""
    narrow = [0.0] * NA
    for i in range(20, 31):
        narrow[i] = 1.0 / 11.0
    m = categorical_projection(9.5, 0.9, narrow, VMIN, VMAX)
    unclipped = 9.5 + 0.9 * distribution_mean(narrow, grid["z"])
    assert abs(distribution_mean(m, grid["z"]) - unclipped) > 0.1
    hi = categorical_projection(50.0, 0.9, grid["p"], VMIN, VMAX)
    assert hi[-1] == pytest.approx(1.0, abs=1e-12)
    lo = categorical_projection(-50.0, 0.9, grid["p"], VMIN, VMAX)
    assert lo[0] == pytest.approx(1.0, abs=1e-12)


def test_the_cross_entropy_is_minimised_at_p_equals_m(grid):
    p = grid["p"]
    entropy = -sum(v * math.log(v) for v in p if v > 0)
    assert categorical_loss(p, p) == pytest.approx(entropy, abs=1e-12)
    assert categorical_loss(p, [1.0 / NA] * NA) > categorical_loss(p, p)


def test_the_greedy_action_maximises_the_expected_return(grid):
    z = grid["z"]
    lo, hi, mid = ([0.0] * NA for _ in range(3))
    lo[10] = hi[40] = mid[25] = 1.0
    a, qs = greedy_action([lo, hi, mid], z)
    assert a == 1
    assert qs[1] == pytest.approx(z[40])
    up = c51_update(0.5, 0.9, [lo, hi, mid], grid["p"], VMIN, VMAX)
    assert up["action"] == 1
    assert up["q_target"] == pytest.approx(0.5 + 0.9 * z[40], abs=1e-9)
    assert up["loss"] == pytest.approx(
        categorical_loss(up["target"], grid["p"]), abs=1e-15)


@pytest.mark.parametrize("gamma", [0.5, 0.8])
def test_the_fixed_point_matches_the_closed_form(grid, gamma):
    """Z = R + gamma Z' with R = +-1 at even odds has E[Z] = 0 and
    Var[Z] = 1/(1 - gamma^2)."""
    dist, info = value_distribution_iteration(
        [-1.0, 1.0], [0.5, 0.5], gamma, VMIN, VMAX, NA, iters=800)
    assert info["converged"]
    assert sum(dist) == pytest.approx(1.0, abs=1e-9)
    mean = distribution_mean(dist, grid["z"])
    var = sum(dist[i] * (grid["z"][i] - mean) ** 2 for i in range(NA))
    assert mean == pytest.approx(0.0, abs=1e-6)
    assert var == pytest.approx(1.0 / (1.0 - gamma * gamma), abs=0.2)


def test_the_bernoulli_alternative(grid):
    b = bernoulli_algorithm(0.0, 1.0, grid["p"], VMIN, VMAX)
    assert 0.0 <= b <= 1.0
    assert bernoulli_algorithm(1000.0, 1.0, grid["p"], VMIN,
                               VMAX) == 1.0


def test_argument_checks(grid):
    p = grid["p"]
    with pytest.raises(ValueError):
        atoms(0.0, 1.0, 1)
    with pytest.raises(ValueError):
        atoms(1.0, 1.0, 5)
    with pytest.raises(ValueError):
        categorical_projection(0.0, 0.9, [0.5] * NA, VMIN, VMAX)
    with pytest.raises(ValueError):
        categorical_projection(0.0, 1.5, p, VMIN, VMAX)
    with pytest.raises(ValueError):
        categorical_projection(0.0, -0.1, p, VMIN, VMAX)
    with pytest.raises(ValueError):
        categorical_loss(p, p[:-1])
    with pytest.raises(ValueError):
        distribution_mean(p[:-1], grid["z"])
    with pytest.raises(ValueError):
        greedy_action([], grid["z"])
    with pytest.raises(ValueError):
        value_distribution_iteration([-1.0, 1.0], [0.5], 0.5, VMIN,
                                     VMAX, NA)
