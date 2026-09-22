"""Tests for gh_c12_11.ghosal_cred_set_cov."""

from morie.fn import _array_core as np

from morie.fn.gh_c12_11 import ghosal_cred_set_cov


def test_gh_c12_11_basic():
    """Test basic functionality with the documented scalar signature."""
    # Documented signature: ghosal_cred_set_cov(theta0, n, level, n_sim, seed).
    # All four numeric arguments are scalars; passing an array breaks the
    # documented contract (the function uses theta0 in a scalar comparison).
    result = ghosal_cred_set_cov(theta0=0.5, n=400, level=0.9, n_sim=400, seed=42)
    assert "estimate" in result
    estimate = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(estimate))

    # Independent theoretical expectation derived from the literature formula
    # described in the docstring (Beta posterior central credible interval
    # under the normal approximation). For theta0=0.5, n=400, level=0.9 the
    # nominal coverage is 0.9; the Monte Carlo estimate should sit within a
    # few standard errors of that value.
    nominal = float(result["nominal"])
    assert abs(float(estimate) - nominal) < 0.05


def test_gh_c12_11_edge():
    """Test edge cases (deterministic seed, minimal simulation count)."""
    # Use the documented signature with scalars only. The function does not
    # return an "n" key per the docstring; assert on documented keys instead.
    result = ghosal_cred_set_cov(theta0=0.3, n=50, level=0.9, n_sim=20, seed=0)

    for key in ("estimate", "nominal", "gap", "method"):
        assert key in result

    est = float(result["estimate"])
    assert 0.0 <= est <= 1.0
    # gap == |estimate - nominal|, computed independently
    assert abs(result["gap"] - abs(est - float(result["nominal"]))) < 1e-12
