"""Tests for gh_c4_13.ghosal_dp_weak_conv."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_13 import ghosal_dp_weak_conv


def test_gh_c4_13_basic():
    """Test basic functionality."""
    G0_A = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha_seq = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
    g = float(np.asarray(G0_A, dtype=float)[0])
    Ms = np.asarray(alpha_seq, dtype=float)
    expected_vars = [g * (1.0 - g) / (1.0 + m) for m in Ms]
    expected_estimate = expected_vars[-1]

    result = ghosal_dp_weak_conv(G0_A, alpha_seq)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert np.isclose(result["estimate"], expected_estimate)
    assert "var_sequence" in result
    assert len(result["var_sequence"]) == len(Ms)
    for v, ev in zip(result["var_sequence"], expected_vars):
        assert np.isclose(v, ev)
    assert "regime" in result
    assert result["regime"] == "DP limit (var positive)"


def test_gh_c4_13_edge():
    """Test edge cases."""
    G0_A = np.array([42.0])
    alpha_seq = np.array([1e-9])
    g = float(np.asarray(G0_A, dtype=float)[0])
    Ms = np.asarray(alpha_seq, dtype=float)
    expected_vars = [g * (1.0 - g) / (1.0 + m) for m in Ms]
    expected_estimate = expected_vars[-1]

    result = ghosal_dp_weak_conv(G0_A, alpha_seq)
    assert "estimate" in result
    assert np.isclose(result["estimate"], expected_estimate)
    assert "n" not in result
    assert result["regime"].startswith("degenerate at random point")
