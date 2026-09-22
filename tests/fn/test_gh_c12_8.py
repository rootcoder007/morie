"""Tests for gh_c12_8.ghosal_cox_bvm_sp."""

from morie.fn import _array_core as np

from morie.fn.gh_c12_8 import ghosal_cox_bvm_sp


def test_gh_c12_8_basic():
    """Test basic functionality."""
    result = ghosal_cox_bvm_sp(beta0=0.8, n=600, seed=42)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c12_8_edge():
    """Test edge cases with a tiny sample."""
    result = ghosal_cox_bvm_sp(beta0=0.8, n=4, seed=1)
    assert "estimate" in result
    # Compute the expected MAP estimate independently using the same
    # partial-likelihood formula on the same inputs.
    import math
    rng = np.random.default_rng(1)
    zs = [1.0 if i % 2 == 0 else 0.0 for i in range(4)]
    times = [-math.log(max(float(rng.uniform(0, 1)), 1e-12))
             / math.exp(0.8 * z) for z in zs]
    order = sorted(range(4), key=lambda i: times[i])
    def neg_pll(b):
        tot = 0.0
        risk = sum(math.exp(b * z) for z in zs)
        for idx in order:
            tot -= b * zs[idx] - math.log(max(risk, 1e-300))
            risk -= math.exp(b * zs[idx])
        return tot
    grid = [0.8 - 1.0 + 2.0 * j / 50 for j in range(51)]
    vals = [neg_pll(b) for b in grid]
    expected_estimate = grid[vals.index(min(vals))]
    assert abs(result["estimate"] - expected_estimate) < 1e-12
