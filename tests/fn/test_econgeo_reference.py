"""econgeo: indices recomputed from their definitions (REAT-exact values checked in tests/cross)."""

import math

import pytest

from morie.fn.econgeo import (
    coagglomeration_index,
    duranton_overman,
    ellison_glaeser,
    herfindahl_index,
    krugman_index,
    location_quotient,
    lq_gini,
)


def test_lq_krugman_gini_herfindahl():
    E = [[10, 30, 60], [50, 20, 30], [5, 5, 90]]
    lq = location_quotient(E)
    assert lq[1][0] == pytest.approx((50 / 100) / (65 / 300), abs=1e-15)
    d = location_quotient(E, method="difference")
    assert d[2][2] == pytest.approx(0.9 - 180 / 300, abs=1e-15)
    # employment-weighted mean of each column's LQs is 1
    for i in range(3):
        assert sum(lq[r][i] * sum(E[r]) for r in range(3)) / 300 == pytest.approx(1.0, abs=1e-14)
    k = krugman_index([10, 30, 60], [30, 30, 40])
    assert (k.K, k.hoover) == pytest.approx((0.4, 0.2), abs=1e-15)
    assert pytest.approx(0.0, abs=1e-15) == krugman_index([1, 2], [2, 4]).K
    with pytest.raises(ValueError):
        krugman_index([1, 2], [1, 2, 3])
    R = sorted([(0.1 / 0.3), 1.0, 0.6 / 0.4])
    m = sum(R) / 3
    assert lq_gini([10, 30, 60], [30, 30, 40]) == pytest.approx(
        2 / (9 * m) * sum((k + 1) * (r - m) for k, r in enumerate(R))
    )
    assert lq_gini([5, 5], [1, 1]) == pytest.approx(0.0, abs=1e-15)
    h = herfindahl_index([50, 30, 20], k=2)
    assert (h.H, h.normalized, h.equivalent_number, h.CR) == pytest.approx((0.38, 0.07, 1 / 0.38, 0.8))


def test_ellison_glaeser_and_coagglomeration():
    emp, reg = [10, 20, 30, 40], ["a", "a", "b", "c"]
    r = ellison_glaeser(emp, reg, [100, 100, 100])
    x = [1 / 3] * 3
    s = [0.3, 0.3, 0.4]
    G = sum((u - v) ** 2 for u, v in zip(s, x))
    H = sum((e / 100) ** 2 for e in emp)
    assert pytest.approx((G, H), abs=1e-15) == (r.G, r.H)
    assert r.gamma == pytest.approx((G - (2 / 3) * H) / ((2 / 3) * (1 - H)), abs=1e-14)
    s2, s3, z4 = 1 / 3, 1 / 9, sum((e / 100) ** 4 for e in emp)
    var = 2 * (H * H * (s2 - 2 * s3 + s2 * s2) - z4 * (s2 - 4 * s3 + 3 * s2 * s2))
    assert r.z == pytest.approx((G - (1 - s2) * H) / math.sqrt(var), abs=1e-12)
    c = coagglomeration_index(
        [10, 20, 30, 40, 25, 25], [1, 1, 1, 2, 2, 2], ["a", "b", "a", "a", "c", "b"], [100, 100, 100]
    )
    g1 = ellison_glaeser([10, 20, 30], ["a", "b", "a"], [100, 100])  # regions a, b only for industry 1
    del g1
    xs = [1 / 3] * 3
    Gp = sum((u - v) ** 2 for u, v in zip([80 / 150, 45 / 150, 25 / 150], xs))
    w = [60 / 150, 90 / 150]
    H1 = (100 + 400 + 900) / 3600
    H2 = (1600 + 625 + 625) / 8100
    G1 = sum((u - v) ** 2 for u, v in zip([40 / 60, 20 / 60, 0.0], xs))
    G2 = sum((u - v) ** 2 for u, v in zip([40 / 90, 25 / 90, 25 / 90], xs))
    ga = [(Gi - (2 / 3) * Hi) / ((2 / 3) * (1 - Hi)) for Gi, Hi in ((G1, H1), (G2, H2))]
    Hc = w[0] ** 2 * H1 + w[1] ** 2 * H2
    want = (Gp / (2 / 3) - Hc - sum(g * wi * wi * (1 - hi) for g, wi, hi in zip(ga, w, (H1, H2)))) / (
        1 - sum(v * v for v in w)
    )
    assert c.gamma_c == pytest.approx(want, abs=1e-14)
    assert c.gamma_i == pytest.approx(ga, abs=1e-14)


def test_duranton_overman():
    P = [(0, 0), (3, 0), (0, 4), (1, 1)]
    k = duranton_overman(P, [0.5, 2.0], bandwidth=0.7)
    d = [3, 4, math.sqrt(2), 5, math.sqrt(5), math.sqrt(10)]
    phi = lambda z: math.exp(-z * z / 2) / math.sqrt(2 * math.pi)  # noqa: E731
    want = [sum(phi((x - v) / 0.7) + phi((x + v) / 0.7) for v in d) / (6 * 0.7) for x in (0.5, 2.0)]
    assert pytest.approx(want, abs=1e-14) == k.K
    k2 = duranton_overman(P, [1.0])
    s = sorted(d)
    sd = math.sqrt(sum((v - sum(d) / 6) ** 2 for v in d) / 5)
    iqr = (s[3] + 0.75 * (s[4] - s[3])) - (s[1] + 0.25 * (s[2] - s[1]))
    assert k2.bandwidth == pytest.approx(0.9 * min(sd, iqr / 1.34) * 6**-0.2, abs=1e-14)
    sites = [(i % 5, i // 5) for i in range(25)]
    b = duranton_overman(P, [1.0, 3.0], sites=sites, n_sim=40, seed=3)
    assert all(lo <= hi for lo, hi in zip(b.lower, b.upper))
    assert b.localized == [kv > u for kv, u in zip(b.K, b.upper)]
