"""Tests for gb1331w.gibbons_wsrt_efficacy."""

import math

from morie.fn import _array_core as np

from morie.fn.gb1331w import gibbons_wsrt_efficacy


def test_gb1331w_basic():
    """Test basic functionality."""
    N = 100
    rng = np.random.default_rng(42)
    f = rng.normal(0, 1, 100)

    # f0 is the standard normal density evaluated at 0.
    f0 = 1.0 / math.sqrt(2.0 * math.pi)

    # integral is the integral of f^2(y) over the real line. For two
    # independent standard normals, the convolution f*f has density
    # N(0, 2); its L2 norm is the reciprocal of the square root of the
    # density at 0 divided by 2*pi... computed directly:
    #   integral of N(0,1)^2 = 1 / (2*sqrt(pi))
    integral = 1.0 / (2.0 * math.sqrt(math.pi))

    result = gibbons_wsrt_efficacy(N, f0, integral)

    assert isinstance(result, dict)

    # Per the docstring, the documented return keys are
    # ``efficacy``, ``limit``, ``integral``, ``n``, ``method``.
    assert "efficacy" in result
    assert "limit" in result
    assert "integral" in result
    assert "n" in result
    assert "method" in result

    # Independent recomputation of the documented formula (13.3.4).
    n_val = float(N)
    f0_val = float(f0)
    i_val = float(integral)
    expected_efficacy = (
        24.0
        * (f0_val / (n_val - 1.0) + i_val) ** 2
        * n_val
        * (n_val - 1.0) ** 2
        / ((n_val + 1.0) * (2.0 * n_val + 1.0))
    )
    expected_limit = 12.0 * n_val * i_val * i_val

    assert math.isclose(result["efficacy"], expected_efficacy, rel_tol=1e-12)
    assert math.isclose(result["limit"], expected_limit, rel_tol=1e-12)
    assert math.isclose(result["integral"], i_val, rel_tol=1e-12)
    assert int(result["n"]) == N


def test_gb1331w_edge():
    """Test edge cases."""
    N = 100
    f0 = 1.0 / math.sqrt(2.0 * math.pi)
    integral = 1.0 / (2.0 * math.sqrt(math.pi))

    result = gibbons_wsrt_efficacy(N, f0, integral)
    assert isinstance(result, dict)
    assert result["n"] == N
    assert math.isfinite(result["efficacy"])
    assert math.isfinite(result["limit"])
