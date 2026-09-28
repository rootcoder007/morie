"""polarize: indices recomputed from their definitions; roll-call utilities on hand-made matrices."""

import math

import pytest

from morie.fn.polarize import (
    earth_movers_distance,
    esteban_ray_index,
    issue_constraint,
    party_divergence,
    party_system_indices,
    polarization_indices,
    polarization_trend,
    rollcall_filter,
    rollcall_matrix,
    rollcall_summary,
)


def test_distributional_indices():
    x = [-2.0, -1.5, -1.0, 0.5, 1.0, 1.8, 2.2]
    r = polarization_indices(x)
    n, m = 7, sum(x) / 7
    d = [v - m for v in x]
    assert r.variance == pytest.approx(sum(v * v for v in d) / 6)
    assert r.gini_mean_difference == pytest.approx(sum(abs(a - b) for a in x for b in x) / 49)
    m2, m3, m4 = (sum(v**k for v in d) / n for k in (2, 3, 4))
    G1 = m3 / m2**1.5 * math.sqrt(n * (n - 1)) / (n - 2)
    G2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * (m4 / m2**2 - 3) + 6)
    assert (r.skewness, r.excess_kurtosis) == pytest.approx((G1, G2))
    assert r.bimodality_coefficient == pytest.approx((G1**2 + 1) / (G2 + 3 * 36 / 20))
    w = polarization_indices([0.0, 1.0], weights=[3.0, 1.0])
    assert w.gini_mean_difference == pytest.approx(2 * 3 * 1 / 16)
    assert esteban_ray_index([0.0, 1.0, 3.0], [0.2, 0.5, 0.3], alpha=1.6) == pytest.approx(
        sum(
            p**2.6 * q * abs(a - b)
            for a, p in zip([0, 1, 3], [0.2, 0.5, 0.3])
            for b, q in zip([0, 1, 3], [0.2, 0.5, 0.3])
        )
    )
    assert earth_movers_distance([0.0, 0.0, 3.0], [1.0]) == pytest.approx((2 / 3) * 1 + (1 / 3) * 2)
    assert earth_movers_distance([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_constraint_trend_party_system():
    X = [[1, 5, 2], [2, 3, 2], [3, 4, 5], [4, 1, 4]]
    c = issue_constraint(X)
    assert c.correlation[0][0] == 1.0 and c.correlation[0][1] == pytest.approx(c.correlation[1][0])
    assert c.constraint == pytest.approx(
        (abs(c.correlation[0][1]) + abs(c.correlation[0][2]) + abs(c.correlation[1][2])) / 3
    )
    t = polarization_trend([1, 2, 3, 4, 5], [1.0, 1.2, 1.1, 1.6, 1.5])
    # mean t = 3, mean y = 1.28, S_ty = 1.40, S_tt = 10
    assert t.slope == pytest.approx(0.14) and t.intercept == pytest.approx(0.86)
    res = [1.0 - 1.00, 1.2 - 1.14, 1.1 - 1.28, 1.6 - 1.42, 1.5 - 1.56]
    assert t.se == pytest.approx(math.sqrt(sum(v * v for v in res) / 3 / 10))
    p = party_system_indices([0.4, 0.4, 0.2])
    assert p.fractionalization == pytest.approx(0.64) and p.effective_number == pytest.approx(1 / 0.36)
    assert p.golosov == pytest.approx(1 + 1 + 0.2 / (0.2 + 0.16 - 0.04))


def test_divergence_and_rollcalls():
    pos = [-0.9, -0.4, 0.2, -0.1, 0.3, 0.8]
    party = ["D", "D", "D", "R", "R", "R"]
    V = [[1, 1, 0], [1, 0, 0], [1, None, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1]]
    d = party_divergence(pos, party, "D", "R", votes=[[float("nan") if v is None else v for v in r] for r in V])
    assert d.mean_difference == pytest.approx(1 / 3 - (-1.1 / 3))
    assert d.median_difference == pytest.approx(0.3 - (-0.4))
    assert d.overlap == 2  # -0.1 and 0.2 lie between min(R) = -0.1 and max(D) = 0.2
    assert d.rice_left == pytest.approx((1 + 0 + 1 / 3) / 3)
    assert d.rice_right == pytest.approx((1 + 1 / 3 + 1) / 3)
    m = rollcall_matrix(["b", "a", "a", "b", "c"], ["v2", "v1", "v2", "v1", "v1"], [1, 5, 2, 0, 9])
    assert m.legislators == ["a", "b", "c"] and m.votes == ["v1", "v2"]
    assert m.matrix[0] == [0.0, 1.0] and math.isnan(m.matrix[1][0]) and m.matrix[1][1] == 1.0
    s = rollcall_summary(V)
    assert (s.yeas, s.nays) == ([3, 2, 4], [3, 3, 2]) and s.participation == [3, 3, 2, 3, 3, 3]
    f = rollcall_filter(V, lop=0.35, minvotes=2)
    assert f.votes == [0, 1] and f.legislators == [0, 1, 3, 4, 5]
