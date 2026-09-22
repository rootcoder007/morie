"""Tests for gh_sup_norm_gp.ghosal_sup_norm_contraction."""

from morie.fn import _array_core as np

from morie.fn.gh_sup_norm_gp import ghosal_sup_norm_contraction


def test_gh_sup_norm_gp_basic():
    """Test basic functionality with documented arguments."""
    s, d, log_power = 1.0, 1.0, 0.5
    ns = (100, 10000, 1000000)
    result = ghosal_sup_norm_contraction(s=s, d=d, log_power=log_power, ns=ns)

    # The result must expose the documented "estimate" key.
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # "rate_by_n" should also be present per the implementation.
    assert "rate_by_n" in result

    # Compute expected rates independently from the documented formula
    # eps_n = n^{-s/(2s+d)} * (log n)^{log_power}.
    expected_rates = [
        float(n) ** (-s / (2.0 * s + d)) * np.log(float(n)) ** log_power
        for n in ns
    ]
    expected_estimate = expected_rates[-1]

    rate_by_n = result["rate_by_n"]
    assert len(rate_by_n) == len(ns)
    for got, want in zip(rate_by_n, expected_rates):
        assert np.isclose(float(got), float(want))

    # Estimate equals the last rate.
    assert np.isclose(float(result["estimate"]), float(expected_estimate))

    # For s=1, d=1, log_power=0.5 the rate sequence is strictly decreasing.
    assert result["decreasing"] is True


def test_gh_sup_norm_gp_edge():
    """Test edge case with a single-element ns tuple."""
    ns = (42.0,)
    result = ghosal_sup_norm_contraction(ns=ns)

    # The function does not return an "n" key; it returns "rate_by_n".
    assert "rate_by_n" in result
    assert len(result["rate_by_n"]) == 1

    # With default s=1, d=1, log_power=0.5, the single rate is
    # 42^{-1/3} * (log 42)^{0.5}.
    expected = float(42.0) ** (-1.0 / 3.0) * np.log(42.0) ** 0.5
    assert np.isclose(float(result["rate_by_n"][0]), float(expected))
    assert np.isclose(float(result["estimate"]), float(expected))

    # A single-element sequence has no pair to compare, so "decreasing"
    # is vacuously True (all() over an empty generator).
    assert result["decreasing"] is True
