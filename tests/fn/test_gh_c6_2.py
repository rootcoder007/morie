"""Tests for gh_c6_2.ghosal_strong_consist."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c6_2 import ghosal_strong_consist


def test_gh_c6_2_basic():
    """Test basic functionality."""
    result = ghosal_strong_consist(theta0=0.6, eps=0.15, n=2000, seed=42)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # Independent computation: simulate one sample path with the same seed
    # and compute the final Beta tail mass outside (theta0 - eps, theta0 + eps).
    theta0 = 0.6
    eps = 0.15
    n = 2000
    seed = 42
    rng = np.random.default_rng(seed)
    S = 0
    for _ in range(1, n + 1):
        S += 1 if float(rng.uniform(0, 1)) < theta0 else 0
    a = 1.0 + S
    b = 1.0 + n - S
    grid = 2000
    expected_mass = 0.0
    for k in range(grid):
        t = (k + 0.5) / grid
        if abs(t - theta0) > eps:
            expected_mass += math.exp(
                math.lgamma(a + b) - math.lgamma(a)
                - math.lgamma(b)
                + (a - 1.0) * math.log(t)
                + (b - 1.0) * math.log(1.0 - t)) / grid

    assert abs(float(result["estimate"]) - expected_mass) < 1e-12
    # Tail mass outside the eps-ball must be in [0, 1]
    assert 0.0 <= float(result["estimate"]) <= 1.0


def test_gh_c6_2_edge():
    """Test edge cases: result keys and documented behaviour."""
    result = ghosal_strong_consist(theta0=0.5, eps=0.3, n=200, seed=7)

    # Documented return keys
    assert "estimate" in result
    assert "path_masses" in result
    assert "method" in result

    # path_masses is the sequence of tail masses along the path,
    # recorded at checkpoints 50, 200, 800, n=200 (so two entries here)
    assert isinstance(result["path_masses"], list)
    assert len(result["path_masses"]) == 2

    # Final estimate must equal the last entry in path_masses
    assert abs(float(result["estimate"])
               - float(result["path_masses"][-1])) < 1e-15

    # Independent recomputation for this small case
    theta0 = 0.5
    eps = 0.3
    n = 200
    seed = 7
    rng = np.random.default_rng(seed)
    S = 0
    for _ in range(1, n + 1):
        S += 1 if float(rng.uniform(0, 1)) < theta0 else 0
    a = 1.0 + S
    b = 1.0 + n - S
    grid = 2000
    expected_mass = 0.0
    for k in range(grid):
        t = (k + 0.5) / grid
        if abs(t - theta0) > eps:
            expected_mass += math.exp(
                math.lgamma(a + b) - math.lgamma(a)
                - math.lgamma(b)
                + (a - 1.0) * math.log(t)
                + (b - 1.0) * math.log(1.0 - t)) / grid

    assert abs(float(result["estimate"]) - expected_mass) < 1e-12
