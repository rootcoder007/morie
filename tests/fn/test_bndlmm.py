"""Tests for bndlmm.bound_linear_min_max."""

from morie.fn import _array_core as np

from morie.fn.bndlmm import bound_linear_min_max


def test_bndlmm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # theta is shape (n, K): K candidate lower bounds per observation
    theta = rng.normal(0, 1, (100, 3))
    # moments is shape (n, J): J candidate upper bounds per observation
    moments = rng.normal(0, 1, (100, 2))
    result = bound_linear_min_max(theta, moments)
    assert isinstance(result, dict)
    # Documented return keys
    for key in ("lower", "upper", "width", "lower_pc", "upper_pc",
                "width_pc", "K", "J", "n"):
        assert key in result

    n, K, J = 100, 3, 2
    assert result["K"] == K
    assert result["J"] == J
    assert result["n"] == n

    # Independent recomputation of the plug-in bounds.
    L = theta
    U = moments
    lo_expected = float(L.mean(axis=0).max())
    hi_expected = float(U.mean(axis=0).min())
    assert result["lower"] == lo_expected
    assert result["upper"] == hi_expected
    assert result["width"] == hi_expected - lo_expected

    # Independent recomputation of the Bonferroni precision-corrected bounds.
    from math import sqrt
    from statistics import NormalDist
    zK = NormalDist().inv_cdf(1.0 - 0.5 / K)
    zJ = NormalDist().inv_cdf(1.0 - 0.5 / J)
    rn = sqrt(n)
    meansL = L.mean(axis=0)
    sdsL = L.std(axis=0, ddof=1)
    meansU = U.mean(axis=0)
    sdsU = U.std(axis=0, ddof=1)
    lo_pc_expected = float((meansL - zK * sdsL / rn).max())
    hi_pc_expected = float((meansU + zJ * sdsU / rn).min())
    assert result["lower_pc"] == lo_pc_expected
    assert result["upper_pc"] == hi_pc_expected
    assert result["width_pc"] == hi_pc_expected - lo_pc_expected


def test_bndlmm_edge():
    """Test edge cases: K = J = 1 reduces to Bonferroni multiplier zero."""
    rng = np.random.default_rng(42)
    theta = rng.normal(0, 1, (50, 1))
    moments = rng.normal(0, 1, (50, 1))
    result = bound_linear_min_max(theta, moments)
    assert isinstance(result, dict)
    assert result["K"] == 1
    assert result["J"] == 1

    L = theta[:, 0]
    U = moments[:, 0]
    lo_expected = float(L.mean())
    hi_expected = float(U.mean())
    assert result["lower"] == lo_expected
    assert result["upper"] == hi_expected
    # With a single candidate, the Bonferroni z multiplier is zero so
    # the precision-corrected bounds coincide with the plug-in bounds.
    assert result["lower_pc"] == lo_expected
    assert result["upper_pc"] == hi_expected
    assert result["width_pc"] == hi_expected - lo_expected
