"""Tests for genvdm.d_study_decision."""

from morie.fn import _array_core as np

from morie.fn.genvdm import d_study_decision


def test_genvdm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # G_components is a tuple of three variances (sigma^2_p, sigma^2_i, sigma^2_pi)
    G_components = (
        rng.normal(0, 1, 100).var(ddof=1),
        rng.normal(0, 1, 100).var(ddof=1),
        rng.normal(0, 1, 100).var(ddof=1),
    )
    # n_proposed must be positive integers (candidate numbers of conditions)
    n_proposed = rng.integers(1, 100, size=100)
    result = d_study_decision(G_components, n_proposed)
    # Documented return is a RichResult with the documented keys
    assert hasattr(result, "e_rho2")
    assert hasattr(result, "phi")
    assert hasattr(result, "meets_target")
    assert hasattr(result, "n_required")
    assert hasattr(result, "n")
    # The coefficients must match the documented formula evaluated on the inputs
    vp, vi, vpi = G_components
    expected_er = [vp / (vp + vpi / k) for k in n_proposed]
    expected_ph = [vp / (vp + (vi + vpi) / k) for k in n_proposed]
    target = 0.8
    expected_meets = [1 if (v == v and v >= target) else 0 for v in expected_er]
    expected_n_required = next(
        (int(n_proposed[i]) for i, m in enumerate(expected_meets) if m == 1),
        0,
    )
    assert list(result.e_rho2) == expected_er
    assert list(result.phi) == expected_ph
    assert list(result.meets_target) == expected_meets
    assert result.n_required == expected_n_required
    assert result.n == len(n_proposed)


def test_genvdm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    G_components = (
        rng.normal(0, 1, 100).var(ddof=1),
        rng.normal(0, 1, 100).var(ddof=1),
        rng.normal(0, 1, 100).var(ddof=1),
    )
    n_proposed = rng.integers(1, 100, size=100)
    result = d_study_decision(G_components, n_proposed)
    assert hasattr(result, "n")
    assert result.n == len(n_proposed)
