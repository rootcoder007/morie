"""Tests for smcopt.sequential_mc."""

from morie.fn.smcopt import sequential_mc


def test_smcopt_basic():
    """Test basic functionality."""

    def objective(x):
        return (x[0] - 1.5) ** 2

    def initial(g):
        return [8.0 * g.random() - 4.0]

    result = sequential_mc(objective, initial)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_smcopt_edge():
    """Test edge cases."""

    def objective(x):
        return (x[0] - 1.5) ** 2

    def initial(g):
        return [8.0 * g.random() - 4.0]

    result = sequential_mc(objective, initial)
    assert isinstance(result, dict)
