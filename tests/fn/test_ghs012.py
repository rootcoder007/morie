"""Tests for ghs012.ghosal_ch3_countable_dirichlet_posterior_l."""

from morie.fn import _array_core as np

from morie.fn.ghs012 import ghosal_ch3_countable_dirichlet_posterior_l


def test_ghs012_basic():
    """Test basic functionality."""
    alpha_j = [1.0, 2.0, 3.0, 4.0, 5.0]
    N_j = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]  # 6 counts so l=3 leaves a tail
    l = 3
    alpha_tail = 100.0
    result = ghosal_ch3_countable_dirichlet_posterior_l(
        alpha_j, N_j, l, alpha_tail
    )
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result

    # Independent computation of the literature formula
    a = list(alpha_j)[:l]
    N = list(N_j)
    n = sum(N)
    expected_upd = [ai + Ni for ai, Ni in zip(a, N[:l])]
    expected_tail = float(alpha_tail) + n - sum(N[:l])
    expected_posterior = expected_upd + [expected_tail]
    expected_estimate = expected_upd[0]

    assert result["estimate"] == expected_estimate
    assert list(result["posterior"]) == expected_posterior


def test_ghs012_edge():
    """Test edge cases: counts exactly length l -> tail = alpha_tail."""
    alpha_j = [0.5, 1.5, 2.5]
    N_j = [2.0, 4.0, 6.0]
    l = 3
    alpha_tail = 7.5
    result = ghosal_ch3_countable_dirichlet_posterior_l(
        alpha_j, N_j, l, alpha_tail
    )
    assert isinstance(result, dict)

    # Independent computation
    a = list(alpha_j)[:l]
    N = list(N_j)
    n = sum(N)
    expected_upd = [ai + Ni for ai, Ni in zip(a, N[:l])]
    expected_tail = float(alpha_tail) + n - sum(N[:l])
    expected_posterior = expected_upd + [expected_tail]

    assert list(result["posterior"]) == expected_posterior
