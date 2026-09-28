"""Tests for morie.fn.wildlife."""

import math

import pytest

from morie.fn import wildlife as W


def test_mcp():
    r = W.mcp_home_range([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (10, 10)], percent=80)
    assert r.area == 4.0 and r.n_used == 5
    assert W.mcp_home_range([(0, 0), (4, 0), (4, 3), (0, 3), (2, 1)], percent=100).area == 12.0
    with pytest.raises(ValueError):
        W.mcp_home_range([(0, 0)] * 4)


def test_distance_sampling_halfnormal_closed_form():
    x = [0.1, 0.4, 0.2, 0.8, 0.5, 0.05, 0.3, 1.1, 0.6, 0.25]
    r = W.distance_sampling(x, width=1e6, effort=2.0)  # no truncation: sigma^2 = mean(x^2)
    assert r.sigma == pytest.approx(math.sqrt(sum(v * v for v in x) / 10), rel=1e-8)
    assert r.esw == pytest.approx(r.sigma * math.sqrt(math.pi / 2), rel=1e-10)
    assert r.density == pytest.approx(10 / (2 * 2.0 * r.esw), rel=1e-12)
    pt = W.distance_sampling(x, width=1e6, transect="point")  # sigma^2 = mean(r^2)/2
    assert pt.sigma == pytest.approx(math.sqrt(sum(v * v for v in x) / 20), rel=1e-8)


def test_occupancy_and_nmixture():
    y = [[1, 0, 1], [0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]]
    r = W.occupancy_model(y)
    assert r.naive_occupancy == 0.5 and 0.5 <= r.occupancy <= 1.0
    assert r.conditional_occupancy[0] == 1.0 and 0 < r.conditional_occupancy[1] < 1
    n = W.nmixture_model([[3, 2, 4], [0, 1, 0], [5, 3, 4], [2, 2, 1]], K=60)
    assert n.K == 60 and n.total_abundance >= 4 * 1.0 and 0 < n.p[0] < 1


def test_connectivity():
    assert W.circuit_resistance([[1, 1, 1]], [(0, 0), (0, 2)]).resistance[0][1] == pytest.approx(2.0)
    par = W.circuit_resistance([[1, 1], [1, 1]], [(0, 0), (0, 1)]).resistance[0][1]
    assert par == pytest.approx(1 / (1 + 1 / 3))  # direct edge in parallel with a 3-edge path
    r = W.least_cost_path([[1, 1, 1], [9, 9, 1], [1, 1, 1]], (0, 0), (2, 0), directions=4)
    assert r.cost == 6.0 and r.path[0] == (0, 0) and r.path[-1] == (2, 0)
    assert r.corridor[0][2] and not r.corridor[1][0]
    assert W.resistance_from_suitability([0.0, 1.0], c=4) == pytest.approx([100.0, 1.0])
    s = [[0.5, 1.0], [0.8, 1.0]]
    assert W.habitat_suitability_index(s)[0] == pytest.approx(math.sqrt(0.4))
    assert W.habitat_suitability_index(s, method="minimum") == [0.5, 1.0]


def test_mantel_geneflow_hanski():
    A = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    C = [[0, 2, 1, 5], [2, 0, 3, 1], [1, 3, 0, 2], [5, 1, 2, 0]]
    assert W.partial_mantel(A, A, C, nsim=0).statistic == pytest.approx(1.0)
    assert W.gene_flow_nm(0.2) == 1.0
    assert W.hanski_connectivity([(0, 0), (1, 0)], [1, 1], [4, 9], alpha=1, b=0.5) == pytest.approx(
        [3 * math.exp(-1), 2 * math.exp(-1)]
    )
