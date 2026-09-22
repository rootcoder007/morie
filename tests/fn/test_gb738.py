"""Tests for gb738.gibbons_chernoff_savage."""

from morie.fn import _array_core as np

from morie.fn.gb738 import gibbons_chernoff_savage


def test_gb738_basic():
    """Test basic functionality."""
    lam = 0.3
    n = 100
    nodes = 201

    def J(u):
        return u

    def Jprime(u):
        return 1.0

    result = gibbons_chernoff_savage(J, Jprime, lam, n, nodes=nodes)

    # Documented return-value keys
    assert isinstance(result, dict)
    for key in ("mean", "var", "sd", "integral", "lam", "n", "method"):
        assert key in result

    # Mean: integral of J(u) du on (0, 1) = integral of u du = 0.5.
    # Independently recomputed via trapezoidal rule on the same grid.
    h = 1.0 / (nodes - 1)
    us = [k * h for k in range(nodes)]
    expected_mean = 0.0
    for k in range(nodes - 1):
        expected_mean += 0.5 * h * (us[k] + us[k + 1])
    assert abs(result["mean"] - expected_mean) < 1e-9

    # For J(u) = u, J'(u) = 1, so cum[k] = integral_0^{u_k} x dx = u_k^2 / 2.
    # Then integral = int_0^1 (1 - u) * (u^2 / 2) du = 1/12.
    # Independently recomputed via trapezoidal rule on the same grid.
    expected_integ = 0.0
    for k in range(nodes - 1):
        u0, u1 = us[k], us[k + 1]
        f0 = (1.0 - u0) * (u0 * u0 / 2.0)
        f1 = (1.0 - u1) * (u1 * u1 / 2.0)
        expected_integ += 0.5 * h * (f0 + f1)
    assert abs(result["integral"] - expected_integ) < 1e-9

    # var = 2 * (1 - lam) * integ / (n * lam), from the formula.
    expected_var = 2.0 * (1.0 - lam) * expected_integ / (n * lam)
    assert abs(result["var"] - expected_var) < 1e-12

    # sd must equal sqrt(var) when var > 0.
    expected_sd = (result["var"]) ** 0.5
    assert abs(result["sd"] - expected_sd) < 1e-12

    # Round-tripped scalars.
    assert result["lam"] == lam
    assert result["n"] == n


def test_gb738_edge():
    """Test edge cases: J(u) = 1, J'(u) = 0 gives zero variance."""
    lam = 0.5
    n = 50

    def J(u):
        return 1.0

    def Jprime(u):
        return 0.0

    result = gibbons_chernoff_savage(J, Jprime, lam, n, nodes=101)

    assert isinstance(result, dict)
    for key in ("mean", "var", "sd", "integral", "lam", "n", "method"):
        assert key in result

    # Mean = integral_0^1 1 du = 1.
    assert abs(result["mean"] - 1.0) < 1e-12

    # J' = 0 everywhere => the double integral (and hence var/sd) is 0.
    assert abs(result["integral"]) < 1e-12
    assert abs(result["var"]) < 1e-12
    assert result["sd"] != result["sd"]  # nan, since sqrt of 0 with var==0 path
