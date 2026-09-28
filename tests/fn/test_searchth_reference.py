"""searchth: Koopman/Frost formulas, their integrals and the optimality conditions."""

import math

import pytest

from morie.fn.searchth import (
    datum_probability_map,
    lateral_range,
    optimal_circular_search,
    optimal_effort_allocation,
    parallel_sweep_pod,
    search_pod,
    search_success,
    sweep_width,
)


def test_lateral_range_models_have_sweep_width_w():
    W = 2.5
    xs = [-60 + 0.001 * k for k in range(120001)]
    for model, kw in (("definite", {}), ("mbeta", {"m": 0.25})):
        assert sweep_width(xs, lateral_range(xs, W, model=model, **kw)) == pytest.approx(W, abs=2e-3)
    # inverse cube: tails decay like W^2/(4 pi x^2); add the analytic tail beyond |x| = 60
    ic = sweep_width(xs, lateral_range(xs, W))
    assert ic + 2 * W * W / (4 * math.pi * 60) == pytest.approx(W, abs=1e-5)
    assert lateral_range([0.0], W) == [1.0]
    with pytest.raises(ValueError):
        lateral_range([1.0], W, model="bogus")


def test_pod_curves_and_parallel_sweeps():
    assert search_pod([1.0, 2.0])[1] == pytest.approx(1 - math.exp(-2))
    assert search_pod([0.4, 1.5], model="definite") == [0.4, 1.0]
    for S in (0.7, 1.3, 3.0):
        assert parallel_sweep_pod(1.0, S) == pytest.approx(math.erf(math.sqrt(math.pi) / (2 * S)), abs=1e-12)
    # definite range sweeps: POD = min(W/S, 1)
    assert parallel_sweep_pod(1.0, 2.0, model="definite", n_points=4001) == pytest.approx(0.5, abs=1e-3)
    # navigation error pushes the inverse cube POD toward random search at equal coverage
    exact = parallel_sweep_pod(1.0, 1.0)
    noisy = parallel_sweep_pod(1.0, 1.0, nav_sd=2.0, n_points=401)
    assert 1 - math.exp(-1) - 1e-3 < noisy < exact


def test_success_bookkeeping_and_datums():
    r = search_success([0.5, 0.3, 0.2], [0.8, 0.5, 0.0])
    assert r.pos == pytest.approx(0.55) and sum(r.poc_updated) == pytest.approx(1.0)
    assert r.poc_updated == pytest.approx([0.1 / 0.45, 0.15 / 0.45, 0.2 / 0.45])
    two = search_success([0.6, 0.4], [[0.5, 0.2], [0.5, 0.2]])
    assert two.cumulative_pod == pytest.approx([0.75, 0.36]) and two.pos == pytest.approx(0.6 * 0.75 + 0.4 * 0.36)
    edges = [-8 + k for k in range(17)]
    pt = datum_probability_map(edges, edges, center=(0.3, -0.2), sigma=1.5)
    assert sum(map(sum, pt.poc)) + pt.outside == pytest.approx(1.0) and pt.outside < 1e-6
    ln = datum_probability_map(edges, edges, datum="line", center=(-3.0, 0.0), end=(3.0, 0.0), sigma=0.5)
    # uniform along the segment: the two middle columns of the central band carry equal mass
    assert ln.poc[8][7] == pytest.approx(ln.poc[8][8], rel=1e-9) and sum(map(sum, ln.poc)) == pytest.approx(
        1.0, abs=1e-9
    )
    ar = datum_probability_map([0, 1, 2], [0, 1, 2], datum="area", center=(0.5, 0.5), end=(1.5, 2.0))
    assert ar.poc == [[1 / 6, 1 / 6], [1 / 3, 1 / 3]]


def test_optimal_allocations():
    r = optimal_circular_search(2.0, 0.5, 30.0)
    R = (4 * 4 * 0.5 * 30 / math.pi) ** 0.25
    u = R * R / 8
    assert (r.radius, r.pos) == pytest.approx((R, 1 - (1 + u) * math.exp(-u)))
    # the effort density integrates to the total effort
    n = 20000
    tot = sum(
        2 * math.pi * rr * (R * R - rr * rr) / (2 * 4 * 0.5) * (R / n) for rr in ((k + 0.5) * R / n for k in range(n))
    )
    assert tot == pytest.approx(30.0, rel=1e-6)
    a = optimal_effort_allocation([0.5, 0.3, 0.15, 0.05], [2.0, 1.0, 1.0, 3.0], 0.8, 4.0)
    assert sum(a.effort) == pytest.approx(4.0, rel=1e-10)
    dens = [p * math.exp(-c) / A for p, c, A in zip([0.5, 0.3, 0.15, 0.05], a.coverage, [2.0, 1.0, 1.0, 3.0])]
    used = [d for d, z in zip(dens, a.effort) if z > 0]
    assert max(used) == pytest.approx(min(used), rel=1e-9)  # posterior densities equalised where searched
    assert all(d <= used[0] * (1 + 1e-9) for d, z in zip(dens, a.effort) if z == 0)
    assert a.pos == pytest.approx(sum(p * (1 - math.exp(-c)) for p, c in zip([0.5, 0.3, 0.15, 0.05], a.coverage)))
