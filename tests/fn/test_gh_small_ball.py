"""Tests for gh_small_ball.ghosal_small_ball_prob."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_small_ball import ghosal_small_ball_prob


def test_gh_small_ball_basic():
    """Test basic functionality with default-style arguments."""
    result = ghosal_small_ball_prob()
    assert "estimate" in result
    assert "phi_by_eps" in result
    assert "method" in result
    # estimate must be a finite scalar (the phi estimate at the smallest eps)
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # phi_by_eps is one value per eps in eps_list (default has 3 entries)
    assert len(result["phi_by_eps"]) == 3
    # Independently verify the increasing flag from the returned phi_by_eps.
    phis = list(result["phi_by_eps"])
    expected_increasing = all(
        phis[i + 1] >= phis[i] - 1e-9 for i in range(len(phis) - 1)
    )
    assert bool(result["increasing"]) == expected_increasing


def test_gh_small_ball_edge():
    """Test with a single-eps, deterministic, very small Monte Carlo run."""
    # eps_list has one entry -> one phi value -> 'estimate' is that value.
    # Use seed for reproducibility; n_grid=8, n_sim=200 keeps the test fast.
    result = ghosal_small_ball_prob(eps_list=(0.5,), n_grid=8, n_sim=200, seed=0)
    assert "estimate" in result
    assert len(result["phi_by_eps"]) == 1
    # For a single eps there are no adjacent pairs, so 'increasing' is True.
    assert bool(result["increasing"]) is True
    # Independent computation: re-run the same BM-small-ball Monte Carlo here
    # using only stdlib + math, and compare the resulting phi estimate.
    n_grid = 8
    n_sim = 200
    eps = 0.5
    seed = 0
    # Match numpy's default_rng(0).normal(0,1) sequence via a simple LCG-free
    # approach: use numpy itself but recompute the scalar from scratch.
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        w = 0.0
        ok = True
        for _ in range(n_grid):
            w += float(rng.normal(0, 1)) / math.sqrt(n_grid)
            if abs(w) >= eps:
                ok = False
                break
        hits += 1 if ok else 0
    expected_estimate = -math.log(max(hits, 1) / n_sim)
    assert abs(float(result["estimate"]) - expected_estimate) < 1e-12
