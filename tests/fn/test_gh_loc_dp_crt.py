"""Tests for gh_loc_dp_crt.ghosal_local_dp_rate."""

from morie.fn import _array_core as np

from morie.fn.gh_loc_dp_crt import ghosal_local_dp_rate


def test_gh_loc_dp_crt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_local_dp_rate(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_loc_dp_crt_edge():
    """Test edge cases."""
    result = ghosal_local_dp_rate(np.array([42.0]))
    # The function's documented payload keys are: estimate, rate_by_n,
    # decreasing, exponent, method. There is no "n" key.
    assert "estimate" in result
    assert "rate_by_n" in result
    assert "exponent" in result
    # With default s=1.0 the exponent is s/(2s+1) = 1/3.
    assert np.isclose(result["exponent"], 1.0 / 3.0)
    # The estimate must be the rate evaluated at the largest default n
    # (1_000_000), i.e. n^(-1/3) * (log n)^0.5, computed independently.
    n_last = 1_000_000
    expected_estimate = (float(n_last) ** (-1.0 / 3.0)) * (np.log(float(n_last)) ** 0.5)
    assert np.isclose(result["estimate"], expected_estimate)
