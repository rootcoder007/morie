"""Tests for gh_c3_10.ghosal_norm_crm."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_10 import ghosal_norm_crm


def test_gh_c3_10_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_norm_crm(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c3_10_edge():
    """Test edge cases."""
    n_jumps = 400
    seed = 42
    result = ghosal_norm_crm(np.array([42.0]), n_jumps=n_jumps, seed=seed)
    assert result["n_jumps"] == n_jumps
    assert result["total_mass"] == 1.0

    # Independent recomputation: mirror the documented formula.
    rng = np.random.default_rng(seed)
    locs = [float(v) for v in rng.uniform(0, 1, n_jumps)._flat()]
    jumps = [float(rng.gamma(1.0 / n_jumps * 4.0, 1.0))
             for _ in range(n_jumps)]
    tot = sum(jumps)
    w = [j / tot for j in jumps]
    half = sum(wi for wi, t in zip(w, locs) if t < 0.5)

    assert np.isclose(float(result["estimate"]), half)
