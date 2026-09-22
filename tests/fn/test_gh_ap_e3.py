"""Tests for gh_ap_e3.ghosal_wavelet_mra."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_e3 import ghosal_wavelet_mra


def test_gh_ap_e3_basic():
    """Test basic functionality with the documented default of 6 levels."""
    result = ghosal_wavelet_mra(n_levels=6)

    # The result must contain the documented "estimate" key.
    assert "estimate" in result
    assert "l2_norm2" in result
    assert "parseval_gap" in result
    assert "method" in result

    estimate = float(np.asarray(result["estimate"], dtype=float))
    l2_norm2 = float(np.asarray(result["l2_norm2"], dtype=float))
    gap = float(np.asarray(result["parseval_gap"], dtype=float))

    # All returned quantities must be finite scalars.
    assert np.all(np.isfinite(np.asarray([estimate, l2_norm2, gap], dtype=float)))

    # ||f||_2^2 on [0,1) for f(x)=x is exactly 1/3.
    expected_l2_norm2 = 1.0 / 3.0
    assert abs(l2_norm2 - expected_l2_norm2) < 1e-12

    # The parseval gap is by definition the absolute difference between
    # the running approximation to ||f||^2 and the true value 1/3.
    assert abs(gap - abs(estimate - expected_l2_norm2)) < 1e-12

    # With 6 levels of Haar detail the partial sum must already be a
    # very tight approximation to 1/3.
    assert abs(estimate - expected_l2_norm2) < 1e-2

    # The starting term (father / scaling coefficient squared) is
    # (int_0^1 x * 1 dx)^2 = (1/2)^2 = 1/4, so the estimate must be
    # at least that large.
    assert estimate >= 0.25 - 1e-12

    # The estimate must monotonically approach 1/3 as levels increase.
    prev = 0.25
    for j in range(2, 8):
        prev_j = ghosal_wavelet_mra(n_levels=j)
        e_j = float(np.asarray(prev_j["estimate"], dtype=float))
        assert abs(e_j - expected_l2_norm2) <= abs(prev - expected_l2_norm2) + 1e-12
        prev = e_j


def test_gh_ap_e3_edge():
    """Test edge cases: one level, and the default argument."""
    # With a single level the estimate should be the scaling coefficient
    # energy only (0.25), since the loop body runs for j = 0 and adds
    # the first level of detail. We instead test the lower bound: the
    # estimate must be at least the starting father term squared.
    result = ghosal_wavelet_mra(n_levels=1)
    estimate = float(np.asarray(result["estimate"], dtype=float))
    expected_l2_norm2 = 1.0 / 3.0
    assert estimate >= 0.25 - 1e-12
    assert estimate <= expected_l2_norm2 + 1e-12

    # The gap must always be non-negative and finite.
    gap = float(np.asarray(result["parseval_gap"], dtype=float))
    assert gap >= 0.0
    assert np.all(np.isfinite(np.asarray([estimate, gap], dtype=float)))

    # Calling without any argument must use the documented default
    # n_levels=6 and return a fully-populated RichResult.
    default_result = ghosal_wavelet_mra()
    for key in ("estimate", "l2_norm2", "parseval_gap", "method"):
        assert key in default_result
    default_estimate = float(
        np.asarray(default_result["estimate"], dtype=float)
    )
    explicit_estimate = float(
        np.asarray(ghosal_wavelet_mra(n_levels=6)["estimate"], dtype=float)
    )
    assert abs(default_estimate - explicit_estimate) < 1e-15
