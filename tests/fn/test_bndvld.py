"""Tests for bndvld.bound_validity_check."""

from morie.fn import _array_core as np

from morie.fn.bndvld import bound_validity_check


def test_bndvld_basic():
    """Test basic functionality."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(43).normal(0, 1, 100)
    # Make sure each upper >= lower so the bounds are individually valid
    upper = upper + np.abs(lower) + 1.0
    theta_0 = 0.0  # scalar, as the function expects
    H0 = 1.0       # scalar
    result = bound_validity_check(lower, upper, theta_0, H0)

    assert isinstance(result, dict)
    # The function returns these documented keys
    for key in ("lower", "upper", "width", "refuted", "covers", "reject", "n", "method"):
        assert key in result

    # Recompute the documented formula independently from the same inputs
    L_expected = float(max(lower))
    U_expected = float(min(upper))
    width_expected = U_expected - L_expected
    refuted_expected = 1.0 if L_expected > U_expected else 0.0
    covers_expected = 1.0 if (L_expected <= theta_0 <= U_expected) else 0.0
    reject_expected = 1.0 if (float(H0) != 0.0 and covers_expected == 0.0) else 0.0
    n_expected = len(lower)

    assert result["lower"] == L_expected
    assert result["upper"] == U_expected
    assert result["width"] == width_expected
    assert result["refuted"] == refuted_expected
    assert result["covers"] == covers_expected
    assert result["reject"] == reject_expected
    assert result["n"] == n_expected


def test_bndvld_edge():
    """Test edge cases."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(43).normal(0, 1, 100)
    upper = upper + np.abs(lower) + 1.0
    theta_0 = 0.0
    H0 = 1.0
    result = bound_validity_check(lower, upper, theta_0, H0)

    assert isinstance(result, dict)
    assert "method" in result

    # When H0 == 0 the function only reports the intersection, leaving
    # ``reject`` at 0 regardless of coverage.
    H0_zero = 0.0
    result_h0_zero = bound_validity_check(lower, upper, theta_0, H0_zero)
    assert result_h0_zero["reject"] == 0.0

    # Empty intersection refutes the maintained assumptions.
    lo_empty = np.array([0.5, 0.7, 0.9])
    hi_empty = np.array([0.1, 0.2, 0.3])
    res_empty = bound_validity_check(lo_empty, hi_empty, 0.5, 1.0)
    assert res_empty["refuted"] == 1.0
    assert res_empty["covers"] == 0.0

    # A non-empty intersection that excludes theta_0 rejects the null.
    lo_excl = np.array([-1.0, -0.5])
    hi_excl = np.array([0.5, 0.4])
    res_excl = bound_validity_check(lo_excl, hi_excl, 10.0, 1.0)
    assert res_excl["refuted"] == 0.0
    assert res_excl["covers"] == 0.0
    assert res_excl["reject"] == 1.0
