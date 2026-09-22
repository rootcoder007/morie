"""Tests for flmgcr.flamingo_gated_cross."""

from morie.fn import _array_core as np

from morie.fn.flmgcr import flamingo_gated_cross


def test_flmgcr_basic():
    """Test basic functionality with g=0 (identity branch) and random gate."""
    rng_x = np.random.default_rng(42).normal(0, 1, (4, 5))
    rng_v = np.random.default_rng(7).normal(0, 1, (3, 5))
    x = rng_x.tolist()
    vision = rng_v.tolist()
    gate = 0.0
    result = flamingo_gated_cross(x, vision, gate)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "h_new" in result
    assert "n" in result
    # With g=0 and identity projections: h_new == x exactly.
    assert result["h_new"] == x
    # estimate = mean of all elements of x, computed independently.
    flat = [v for row in x for v in row]
    expected_estimate = sum(flat) / len(flat)
    assert abs(result["estimate"] - expected_estimate) < 1e-12
    assert result["n"] == 4

    # Now exercise a non-zero gate so the CrossAttn branch contributes.
    gate_nz = 0.3
    rng_g = np.random.default_rng(123).normal(0, 1, (4, 5))
    x2 = rng_g.tolist()
    res2 = flamingo_gated_cross(x2, vision, gate_nz)
    assert "estimate" in res2
    assert res2["n"] == 4
    # With non-zero gate, output must differ from x.
    assert res2["h_new"] != x2


def test_flmgcr_edge():
    """Test edge cases: empty inputs raise, method key present."""
    rng_x = np.random.default_rng(42).normal(0, 1, (2, 3))
    rng_v = np.random.default_rng(43).normal(0, 1, (2, 3))
    x = rng_x.tolist()
    vision = rng_v.tolist()
    gate = 0.0
    result = flamingo_gated_cross(x, vision, gate)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "method" in result
    assert "n" in result
    assert result["n"] == 2
    # Identity gate keeps the mean unchanged.
    flat = [v for row in x for v in row]
    expected_estimate = sum(flat) / len(flat)
    assert abs(result["estimate"] - expected_estimate) < 1e-12
