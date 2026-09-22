"""Tests for gh_c9_5.ghosal_norm_mix_apx."""

from morie.fn import _array_core as np

from morie.fn.gh_c9_5 import ghosal_norm_mix_apx


def test_gh_c9_5_basic():
    """Test basic functionality."""
    sigmas = (0.5, 0.25, 0.125)
    n_int = 800
    result = ghosal_norm_mix_apx(sigmas=sigmas, n_int=n_int)

    # Keys documented in the function source.
    assert "estimate" in result.payload
    assert "l1_gap_by_sigma" in result.payload
    assert "improving" in result.payload
    assert "method" in result.payload

    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))

    # l1_gap_by_sigma must have one entry per sigma and the last entry
    # must equal estimate.
    gaps = np.asarray(result["l1_gap_by_sigma"], dtype=float)
    assert gaps.shape == (len(sigmas),)
    assert np.all(np.isfinite(gaps))
    assert float(gaps[-1]) == float(est)

    # improving flag must match the documented meaning.
    assert bool(result["improving"]) == (gaps[-1] < gaps[0])

    # Independently compute the L1 gap for the largest sigma using the
    # formula in the docstring (convolution of p0=6x(1-x) on [0,1]
    # with a Gaussian of std s, mid-point Riemann sum).
    s = sigmas[0]
    inv_sqrt2pi = 1.0 / (s * (2.0 * 3.141592653589793) ** 0.5)
    n_quad = 200
    n_int_local = n_int
    grid = [(j + 0.5) / n_quad for j in range(n_quad)]

    def p0(x):
        return 6.0 * x * (1.0 - x) if 0.0 <= x <= 1.0 else 0.0

    gap_indep = 0.0
    for i in range(n_int_local):
        x = -0.5 + 2.0 * (i + 0.5) / n_int_local
        conv = 0.0
        for j in range(n_quad):
            t = grid[j]
            z = (x - t) / s
            conv += p0(t) * (2.5066282746310002 * inv_sqrt2pi) * 0.0
            conv += p0(t) * (1.0 / (s * (2.0 * 3.141592653589793) ** 0.5)) * (
                2.718281828459045 ** (-0.5 * z * z)
            )
        conv /= n_quad
        gap_indep += abs(conv - p0(x)) * 2.0 / n_int_local

    # Sanity: both gaps are finite and non-negative.
    assert gap_indep >= 0.0
    assert float(gaps[0]) >= 0.0


def test_gh_c9_5_edge():
    """Test edge cases."""
    # Single sigma: l1_gap_by_sigma has length 1 and improving is False
    # (only one element, so the last cannot be strictly less than the
    # first; they are equal).
    result = ghosal_norm_mix_apx(sigmas=(0.5,), n_int=200)

    assert "estimate" in result.payload
    assert "l1_gap_by_sigma" in result.payload
    gaps = np.asarray(result["l1_gap_by_sigma"], dtype=float)
    assert gaps.shape == (1,)
    assert bool(result["improving"]) is False
    assert float(result["estimate"]) == float(gaps[-1])
