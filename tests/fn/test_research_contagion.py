import math

import pytest

from morie.fn.research_contagion import contagion_branching


@pytest.mark.parametrize("n", [0.0, 0.1, 0.5, 0.9, 0.99])
def test_subcritical_cluster_size_rate_and_share(n):
    b = contagion_branching(n, mu=3)
    assert b.subcritical
    partial = math.fsum(n**k for k in range(20000))
    assert b.expected_cluster_size == pytest.approx(partial, rel=1e-9)
    assert b.stationary_rate == pytest.approx(3 * b.expected_cluster_size, rel=1e-12)
    assert (b.stationary_rate - 3) / b.stationary_rate == pytest.approx(n, abs=1e-12)
    assert b.endogeneity_share == n
    assert b.generation_means == [n**k for k in range(10)]


def test_generation_means_follow_the_recursion():
    b = contagion_branching(0.73, mu=2.5, generations=7)
    for k in range(1, 7):
        assert b.generation_means[k] == pytest.approx(0.73 * b.generation_means[k - 1], rel=1e-14)


@pytest.mark.parametrize("n", [1.0, 1.1, 2.0])
def test_critical_and_supercritical(n):
    b = contagion_branching(n)
    assert not b.subcritical
    assert b.expected_cluster_size == math.inf and b.stationary_rate == math.inf
    assert math.isnan(b.endogeneity_share)
    assert all(v >= 1 for v in b.generation_means)


def test_argument_checks():
    with pytest.raises(ValueError, match="non-negative"):
        contagion_branching(-0.1)
    with pytest.raises(ValueError, match="positive"):
        contagion_branching(0.5, mu=0)
