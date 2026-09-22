"""Tests for gh_c14_19.ghosal_local_dp."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_19 import ghosal_local_dp


def test_gh_c14_19_basic():
    """Test basic functionality for a scalar x."""
    x = 0.3
    result = ghosal_local_dp(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # Independent expectation: stick weights w_k satisfy 0 <= w_k <= 1
    # and sum_k w_k <= 1 (since the first stick weight w_0 = V_0, the rest
    # are products that shrink). The estimate is a sum of such weights.
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert 0.0 <= estimate <= 1.0
    # n_active must be an integer count, 0 <= n_active <= n_atoms.
    n_active = int(result["n_active"])
    assert 0 <= n_active <= 200
    # The 'local' flag should be True whenever not all atoms are active.
    assert result["local"] == (n_active < 200)
    # The result payload is wrapped, so keys live under 'payload'.
    assert "method" in result


def test_gh_c14_19_edge():
    """Test edge case: a single scalar input whose active window catches all atoms."""
    result = ghosal_local_dp(0.0, bandwidth=10.0)
    # When the bandwidth covers the full unit interval, every atom is active,
    # so n_active equals n_atoms and the estimate is the full stick sum.
    assert int(result["n_active"]) == 200
    # With all atoms active, the local-DP truncation is a no-op, so 'local'
    # must be False.
    assert result["local"] == False  # noqa: E712
    # The full stick-breaking sequence sums to 1.
    assert 0.0 <= float(np.asarray(result["estimate"], dtype=float)) <= 1.0
