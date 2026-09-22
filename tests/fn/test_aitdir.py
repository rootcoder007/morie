"""Tests for aitdir.dirichlet_density."""

import math

from morie.fn import _array_core as np

from morie.fn.aitdir import dirichlet_density


def test_aitdir_basic():
    """Test basic functionality with a 100-dim simplex point and matching alpha."""
    rng = np.random.default_rng(42)
    # Build a strictly-positive point on the open simplex of dimension 100.
    raw = rng.normal(0.0, 1.0, 100)
    x = [abs(v) + 1e-3 for v in raw]
    s = 0.0
    for v in x:
        s += v
    x = [v / s for v in x]

    # alpha must have the same length as x and be strictly positive.
    alpha = [0.05] * 100

    result = dirichlet_density(x, alpha)

    # The function returns a RichResult (dict-like) with these documented keys.
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "f" in result
    assert "log_f" in result
    assert "log_const" in result
    assert "alpha0" in result
    assert "D" in result
    assert result["D"] == 100

    # Independent computation of the Dirichlet density using plain arithmetic.
    a0 = 0.0
    for a in alpha:
        a0 += a
    log_const = math.lgamma(a0)
    for a in alpha:
        log_const -= math.lgamma(a)
    log_f = log_const
    for i in range(100):
        log_f += (alpha[i] - 1.0) * math.log(x[i])
    expected_f = math.exp(log_f)

    assert math.isclose(result["estimate"], expected_f, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(result["f"], expected_f, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(result["log_f"], log_f, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(result["log_const"], log_const, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(result["alpha0"], a0, rel_tol=1e-12, abs_tol=1e-15)


def test_aitdir_edge():
    """Test edge case: minimum simplex dimension D=2."""
    x = [0.3, 0.7]
    alpha = [1.0, 2.0]

    result = dirichlet_density(x, alpha)

    assert isinstance(result, dict)
    assert result["D"] == 2
    assert "estimate" in result
    assert "log_f" in result

    # Independent computation for D=2.
    a0 = alpha[0] + alpha[1]
    log_const = math.lgamma(a0) - math.lgamma(alpha[0]) - math.lgamma(alpha[1])
    log_f = log_const + (alpha[0] - 1.0) * math.log(x[0]) + (alpha[1] - 1.0) * math.log(x[1])
    expected_f = math.exp(log_f)

    assert math.isclose(result["estimate"], expected_f, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(result["log_f"], log_f, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(result["alpha0"], a0, rel_tol=1e-12, abs_tol=1e-15)
