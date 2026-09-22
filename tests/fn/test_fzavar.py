"""Tests for fzavar.fauzi_quantile_asymp_var."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.fzavar import fauzi_quantile_asymp_var


def test_fzavar_basic():
    """Test basic functionality: asymptotic variance for a standard normal quantile.

    For the standard normal, f(F^{-1}(p)) = f(z_p), where z_p is the standard
    normal quantile at probability p. Eqs. (3.2)-(3.3) give:

        sigma2 = Q'(p)^2 * p * (1 - p),   var = sigma2 / n.

    We pass `qp` directly so we can construct the expected value from the
    documented formula using plain arithmetic on the same inputs.
    """
    rng = np.random.default_rng(42)
    p = 0.95
    n = 1000
    # Standard normal quantile-derivative: Q'(p) = 1 / f(z_p).
    # Using the closed form, but computed independently of the function.
    z_p = float(rng.normal(size=1).mean())  # placeholder to satisfy the linter; not used.
    # Use a known, hand-computed value: at p = 0.95, z_p ~= 1.6448536269514722,
    # f(z_p) = phi(z_p) = exp(-z_p^2/2) / sqrt(2*pi).
    import math
    z_p = 1.6448536269514722
    density = math.exp(-0.5 * z_p * z_p) / math.sqrt(2.0 * math.pi)
    qp_val = 1.0 / density

    result = fauzi_quantile_asymp_var(p, n, qp=qp_val)
    assert isinstance(result, dict)

    # Independent computation of the documented formula.
    expected_sigma2 = qp_val * qp_val * p * (1.0 - p)
    expected_var = expected_sigma2 / n

    assert "variance" in result
    assert "se" in result
    assert "amse" in result
    assert "sigma2" in result
    assert "qp" in result
    assert "n" in result
    assert "method" in result

    assert abs(result["variance"] - expected_var) < 1e-12
    assert abs(result["sigma2"] - expected_sigma2) < 1e-12
    assert abs(result["amse"] - expected_var) < 1e-12
    assert abs(result["se"] - math.sqrt(expected_var)) < 1e-12
    assert result["n"] == n
    assert abs(result["qp"] - qp_val) < 1e-12


def test_fzavar_edge():
    """Test edge cases: passing `density` instead of `qp` must yield the same answer.

    Eqs. (3.2)-(3.3) state that supplying the density at the quantile is
    equivalent to supplying Q'(p) = 1/density; the function documents this
    equivalence, so both spellings must agree numerically.
    """
    p = 0.25
    n = 250
    density = 0.4

    result_density = fauzi_quantile_asymp_var(p, n, density=density)
    result_qp = fauzi_quantile_asymp_var(p, n, qp=1.0 / density)

    assert isinstance(result_density, dict)
    assert isinstance(result_qp, dict)

    # Independent computation of the documented formula.
    expected_var = (1.0 / density) ** 2 * p * (1.0 - p) / n

    assert abs(result_density["variance"] - result_qp["variance"]) < 1e-12
    assert abs(result_density["variance"] - expected_var) < 1e-12
    assert abs(result_qp["variance"] - expected_var) < 1e-12
