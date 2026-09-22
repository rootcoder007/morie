"""Tests for gh_c14_24.ghosal_ibp_stickbr."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_24 import ghosal_ibp_stickbr


def test_gh_c14_24_basic():
    """Test basic functionality."""
    result = ghosal_ibp_stickbr()
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(estimate)

    pi_head = result["pi_head"]
    assert len(pi_head) == 8

    # Compute expected probability of the first feature directly:
    # pi_k = prod_{j<=k} V_j with V_j ~ Beta(alpha, 1).
    # For alpha=2.0, seed=42: simulate the same stick-breaking chain
    # via cumulative product and compare to the returned pi_head.
    rng = np.random.default_rng(42)
    vs = [float(rng.beta(2.0, 1.0)) for _ in range(8)]
    expected_pi_head = []
    cur = 1.0
    for v in vs:
        cur *= v
        expected_pi_head.append(cur)
    for got, exp in zip(pi_head, expected_pi_head):
        assert abs(float(got) - exp) < 1e-12

    # Decay flag is expected to be True for this seed.
    assert result["decreasing"] is True

    # Expected number of features under the IBP prior equals alpha.
    assert result["expected_sum"] == 2.0

    # Method string is documented.
    assert "IBP stick-breaking" in result["method"]


def test_gh_c14_24_edge():
    """Test edge cases."""
    # Single-feature truncation.
    result = ghosal_ibp_stickbr(n_feats=1, alpha=3.5, seed=7)
    assert "estimate" in result
    pi_head = result["pi_head"]
    assert len(pi_head) == 1

    # Independent recomputation for n_feats=1.
    rng = np.random.default_rng(7)
    expected = float(rng.beta(3.5, 1.0))
    assert abs(float(pi_head[0]) - expected) < 1e-12
    assert abs(float(result["estimate"]) - expected) < 1e-12
    assert result["decreasing"] is True
    assert result["expected_sum"] == 3.5
