"""Tests for morie.fn.dplapl -- differential privacy Laplace mechanism."""

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.dplapl import dp_laplace, dplapl


class TestDplapl:
    def test_alias(self):
        assert dplapl is dp_laplace

    def test_mean_query(self):
        x = np.arange(100, dtype=float)
        r = dp_laplace(x, epsilon=10.0, query="mean")
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value - r.extra["true_value"]) < 5

    def test_high_privacy(self):
        x = np.arange(1, 101, dtype=float)
        r1 = dp_laplace(x, epsilon=0.01)
        r2 = dp_laplace(x, epsilon=100.0)
        assert r1.extra["scale"] > r2.extra["scale"]


def test_laplace_noise_replayed():
    """value = true query + Laplace(sensitivity / epsilon) draw from the seed."""
    import pytest

    from morie.fn import _array_core as np

    x = [3.0, 7.0, 1.0, 9.0, 5.0]
    sens = (9.0 - 1.0) / 5
    noise = float(np.random.default_rng(4).laplace(0, sens / 2.0))
    r = dp_laplace(x, epsilon=2.0, query="mean", seed=4)
    assert r.extra["scale"] == pytest.approx(sens / 2.0, rel=1e-15)
    assert r.value == pytest.approx(5.0 + noise, rel=1e-14)
    assert dp_laplace(x, epsilon=1.0, query="count", seed=4).extra["sensitivity"] == 1.0
