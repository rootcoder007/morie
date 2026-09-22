"""Tests for clstrs.cluster_design."""

from morie.fn import _array_core as np

from morie.fn.clstrs import cluster_design


def test_clstrs_basic():
    """Test basic functionality."""
    rho = 0.05
    S2 = 4.0
    c1 = 50.0
    c2 = 1.0
    budget = 10000.0
    result = cluster_design(rho, S2, c1, c2, budget)
    assert isinstance(result, dict)
    for key in ("k_opt", "k", "m", "variance", "deff", "cost", "elements"):
        assert key in result

    # Independent recomputation of the documented formula
    kopt = (c1 * (1.0 - rho) / (c2 * rho)) ** 0.5
    k_floor = max(1, int(kopt // 1))
    candidates = [k_floor, k_floor + 1]

    def V(kk):
        m = budget / (c1 + c2 * kk)
        return (S2 / (m * kk)) * (1.0 + (kk - 1.0) * rho)

    vs = [V(k) for k in candidates]
    k_best = candidates[0] if vs[0] <= vs[1] else candidates[1]
    m_best = budget / (c1 + c2 * k_best)
    var_best = V(k_best)
    deff_best = 1.0 + (k_best - 1.0) * rho
    cost_best = m_best * (c1 + c2 * k_best)
    elems_best = m_best * k_best

    assert result["k_opt"] == kopt
    assert result["k"] == float(k_best)
    assert result["m"] == m_best
    assert result["variance"] == var_best
    assert result["deff"] == deff_best
    assert result["cost"] == cost_best
    assert result["elements"] == elems_best


def test_clstrs_edge():
    """Test edge cases."""
    # rho at its upper bound: k_opt collapses to 1.0
    rho = 1.0
    S2 = 1.0
    c1 = 10.0
    c2 = 1.0
    budget = 1000.0
    result = cluster_design(rho, S2, c1, c2, budget)
    assert isinstance(result, dict)
    assert result["k_opt"] == 1.0
    assert result["k"] in (1.0, 2.0)

    # Independent recomputation
    kopt = 1.0
    k_floor = max(1, int(kopt))
    candidates = [k_floor, k_floor + 1]

    def V(kk):
        m = budget / (c1 + c2 * kk)
        return (S2 / (m * kk)) * (1.0 + (kk - 1.0) * rho)

    vs = [V(k) for k in candidates]
    k_best = candidates[0] if vs[0] <= vs[1] else candidates[1]
    m_best = budget / (c1 + c2 * k_best)
    var_best = V(k_best)

    assert result["m"] == m_best
    assert result["variance"] == var_best
