"""
Tests for Newton-Raphson root finding.
"""

from morie.fn import _array_core as np
from morie.fn.nwtrp import nwtrp


class TestNwtrp:
    """Newton-Raphson root finding tests."""

    def test_nwtrp_quadratic(self):
        """Test finding root of x^2 - 2 = 0."""

        def f(x):
            return x**2 - 2

        def fprime(x):
            return 2 * x

        root = nwtrp(f, fprime, 1.5)
        assert np.isclose(root, np.sqrt(2), atol=1e-6)

    def test_nwtrp_cubic(self):
        """Test finding root of x^3 - 8 = 0."""

        def f(x):
            return x**3 - 8

        def fprime(x):
            return 3 * x**2

        root = nwtrp(f, fprime, 2.5)
        assert np.isclose(root, 2.0, atol=1e-5)

    def test_nwtrp_trig(self):
        """Test finding root of sin(x) near pi."""

        def f(x):
            return np.sin(x)

        def fprime(x):
            return np.cos(x)

        root = nwtrp(f, fprime, 3.0)
        assert np.isclose(root, np.pi, atol=1e-5)

    def test_nwtrp_full_output(self):
        """Test full_output flag."""

        def f(x):
            return x**2 - 1

        def fprime(x):
            return 2 * x

        root, info = nwtrp(f, fprime, 0.5, full_output=True)
        assert "iterations" in info
        assert "converged" in info
        assert "final_residual" in info
        assert np.isclose(root, 1.0, atol=1e-6)

    def test_nwtrp_max_iterations(self):
        """Test convergence with max_iter limit."""

        def f(x):
            return x**2 - 2

        def fprime(x):
            return 2 * x

        root, info = nwtrp(f, fprime, 1.5, max_iter=5, full_output=True)
        assert info["iterations"] <= 5

    def test_nwtrp_multivariate(self):
        """Test multivariate system."""

        def f(x):
            return np.array([x[0] ** 2 - 1, x[1] ** 2 - 4])

        def fprime(x):
            return np.array([[2 * x[0], 0], [0, 2 * x[1]]])

        x0 = np.array([0.5, 1.5])
        root = nwtrp(f, fprime, x0)
        assert np.allclose(root, np.array([1, 2]), atol=1e-5)
