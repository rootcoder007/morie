"""Tests for gh_c11_12.ghosal_selfsim_gp."""

from morie.fn import _array_core as np

from morie.fn.gh_c11_12 import ghosal_selfsim_gp


def test_gh_c11_12_basic():
    """Test basic functionality against the documented self-similarity formula.

    For fractional Brownian motion with Hurst exponent H, the variance at
    scaled time lam*t satisfies
        E[f(lam*t)^2] = lam^(2H) * E[f(t)^2]
    so the ratio of the two variances equals lam^(2H).
    """
    H = 0.6
    lam = 3.0
    t = 0.2
    result = ghosal_selfsim_gp(H=H, lam=lam, t=t)

    assert "estimate" in result
    assert "expected" in result
    assert "gap" in result
    assert "method" in result

    estimate = float(result["estimate"])
    expected = float(result["expected"])
    gap = float(result["gap"])

    # Independent reference computation from the documented formula.
    ref_estimate = (lam * t) ** (2 * H) / (t ** (2 * H))
    ref_expected = lam ** (2 * H)

    assert np.all(np.isfinite(np.asarray(estimate, dtype=float)))
    assert estimate == ref_estimate
    assert expected == ref_expected
    assert gap == abs(ref_estimate - ref_expected)
    assert gap < 1e-12


def test_gh_c11_12_edge():
    """Test edge case: t=1 with H=0.5 (standard Brownian motion self-similarity).

    For H=0.5 the scaling factor is exactly lam (no H exponent effect on the
    ratio beyond the trivial one), so the expected ratio is lam.
    """
    result = ghosal_selfsim_gp(H=0.5, lam=4.0, t=1.0)

    assert "estimate" in result
    assert "expected" in result
    assert "gap" in result

    estimate = float(result["estimate"])
    expected = float(result["expected"])
    gap = float(result["gap"])

    ref_estimate = (4.0 * 1.0) ** (2 * 0.5) / (1.0 ** (2 * 0.5))
    ref_expected = 4.0 ** (2 * 0.5)

    assert estimate == ref_estimate
    assert expected == ref_expected
    assert gap == abs(ref_estimate - ref_expected)
