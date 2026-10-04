"""
Tests for Nesterov accelerated gradient.
"""

from morie.fn import _array_core as np
from morie.fn.nadmf import nadmf


class TestNadmf:
    """NAG tests."""

    def test_nadmf_quadratic(self):
        """Test NAG on 2D quadratic."""

        def f(x):
            return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

        def gf(x):
            return np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])

        x0 = np.array([0.0, 0.0])
        x_min = nadmf(f, gf, x0, learning_rate=0.1)
        assert np.allclose(x_min, [1, 2], atol=1e-2)

    def test_nadmf_rosenbrock(self):
        """Test NAG on Rosenbrock."""

        def f(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        def gf(x):
            return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])

        x0 = np.array([-1.0, -1.0])
        x_min = nadmf(f, gf, x0, learning_rate=0.001, max_iter=5000)
        assert np.allclose(x_min, [1, 1], atol=0.2)

    def test_nadmf_1d(self):
        """Test NAG on 1D."""

        def f(x):
            return x[0] ** 2 - 4 * x[0]

        def gf(x):
            return np.array([2 * x[0] - 4])

        x0 = np.array([0.0])
        x_min = nadmf(f, gf, x0, learning_rate=0.1)
        assert np.isclose(x_min[0], 2.0, atol=1e-2)

    def test_nadmf_momentum_effect(self):
        """Test different momentum values."""

        def f(x):
            return np.sum(x**2)

        def gf(x):
            return 2 * x

        x0 = np.array([5.0, 5.0])

        x_low = nadmf(f, gf, x0, momentum=0.1, max_iter=1000)
        x_high = nadmf(f, gf, x0, momentum=0.99, max_iter=1000)

        # Both should reach near 0, but convergence might differ
        assert np.allclose(x_low, [0, 0], atol=1.0)
        assert np.allclose(x_high, [0, 0], atol=1.0)

    def test_nadmf_full_output(self):
        """Test full_output flag."""

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def gf(x):
            return 2 * x

        x0 = np.array([1.0, 1.0])
        x_min, info = nadmf(f, gf, x0, full_output=True)
        assert "iterations" in info
        assert "converged" in info
        assert "final_value" in info
