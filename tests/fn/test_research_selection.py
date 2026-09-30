import math

import pytest

from morie.fn.research_selection import (
    age_crime_aggregate,
    collider_arrest,
    deterrence_design_check,
    deterrence_response,
    disparity_benchmark,
    disparity_exposure_bounds,
    interracial_rates,
    probability_of_necessity,
    relative_risk_from_or,
)


def test_disparity_exposure_bounds():
    y = {"A": 310, "B": 97, "C": 55}
    m = {"B": 1210, "A": 1000, "C": 400}
    g = 1.35
    d = disparity_exposure_bounds(y, m, gamma=g, reference="B")
    assert d["group"] == ["A", "B", "C"] and d["m"] == [1000.0, 1210.0, 400.0]
    ref = 97 / 1210
    for k, grp in enumerate(d["group"]):
        pr = y[grp] / m[grp]
        assert d["rate_lower"][k] == pytest.approx(pr / g, rel=1e-15)
        assert d["ratio_upper"][k] == pytest.approx(g * g * pr / ref, rel=1e-14)
        assert d["direction_identified"][k] == (pr / ref > g * g or pr / ref < 1 / (g * g))
    with pytest.raises(ValueError, match=">= 1"):
        disparity_exposure_bounds(y, m, gamma=0.5)


def test_age_crime_aggregate_is_the_mixture():
    ages = list(range(10, 40))
    c1 = [math.exp(-((a - 17) ** 2) / 8) for a in ages]
    c2 = [math.exp(-((a - 24) ** 2) / 30) for a in ages]
    r = age_crime_aggregate([0.3, 0.7], {"early": c1, "late": c2}, ages)
    assert r["age"] == ages and r["early"] == c1
    assert r["aggregate"] == pytest.approx([0.3 * a + 0.7 * b for a, b in zip(c1, c2)], abs=1e-15)
    with pytest.raises(ValueError, match="sum to 1"):
        age_crime_aggregate([0.3, 0.3], [c1, c2])


def test_deterrence_design_rank():
    d = deterrence_design_check([0.1, 0.3, 0.5], [2, 2, 2], [30, 30, 30])
    assert d.rank == 2 and d.identified == {"certainty": True, "severity": False, "celerity": False}
    assert d.constant == {"certainty": False, "severity": True, "celerity": True}
    full = deterrence_design_check([0, 1, 0, 0, 0.5], [0, 0, 1, 0, 2], [0, 0, 0, 1, 3])
    assert full.rank == 4 and all(full.identified.values())
    # severity collinear with certainty: neither separately identified, celerity is
    col = deterrence_design_check([0.1, 0.2, 0.3], [2, 4, 6], [1, 1, 2])
    assert col.rank == 3 and col.identified == {"certainty": False, "severity": False, "celerity": True}


def test_disparity_benchmark_product_identity():
    pop, con, force = {"A": 1234, "B": 5678, "C": 910}, {"A": 301, "B": 402, "C": 77}, {"A": 23, "B": 17, "C": 9}
    k = {"A": 1.3, "B": 0.9, "C": 1.1}
    b = disparity_benchmark(pop, con, force, "B", k)
    for i, g in enumerate(b["group"]):
        cd = (con[g] / pop[g]) / (con["B"] / pop["B"])
        fd = (force[g] / con[g]) / (force["B"] / con["B"])
        assert b["resident_disparity"][i] == pytest.approx(cd * fd, rel=1e-14)
        assert abs(b["product_check"][i]) < 1e-12
        assert b["log_shift"][i] == pytest.approx(-math.log(k[g]), abs=1e-15)
        assert b["resident_disparity_corrected"][i] == pytest.approx(cd * fd * k["B"] / k[g], rel=1e-14)


def test_relative_risk_from_odds_ratio():
    assert relative_risk_from_or(0.45).rr_bounds == {"lower": 0.45, "upper": 1}
    q, s, orr = 0.137, 0.41, 3.7
    r = relative_risk_from_or(orr, q, s)
    a, b = r.risks["exposed"], r.risks["unexposed"]
    assert s * a + (1 - s) * b == pytest.approx(q, abs=1e-14)
    assert (a / (1 - a)) / (b / (1 - b)) == pytest.approx(orr, rel=1e-12)
    assert 1 <= r.risks["relative_risk"] <= orr
    assert r.overstatement_factor == pytest.approx(orr / (a / b), rel=1e-14)


def test_deterrence_response_is_the_largest_maximiser():
    xs = list(range(13))
    ben = [3 * v**0.5 - (v % 3 == 0) + 0.1 * math.sin(v) for v in xs]
    san = [v**1.5 + 0.2 * v for v in xs]
    ps = [0, 0.05, 0.1, 0.3, 0.5, 1, 2]
    r = deterrence_response(xs[::-1], ben[::-1], san[::-1], ps)
    for k, p in enumerate(ps):
        u = [b - p * s for b, s in zip(ben, san)]
        assert r["value"][k] == max(u)
        assert r["x_opt"][k] == max(x for x, v in zip(xs, u) if v >= max(u) - 1e-12)
    assert all(r["x_opt"][k] >= r["x_opt"][k + 1] for k in range(len(ps) - 1))
    with pytest.raises(ValueError, match="strictly increasing"):
        deterrence_response([0, 1], [1, 2], [1, 1], [0.5])


def test_interracial_rates_against_the_null():
    off = {"A_on_B": 121, "B_on_A": 203, "A_on_A": 907, "B_on_B": 311, "C_on_A": 40, "A_on_C": 17}
    pop = {"A": 81234, "B": 19876, "C": 5400}
    r = interracial_rates(off, pop)
    N = sum(pop.values())
    pair = [pop[a] / N * pop[b] / N * N for a, b in zip(r["offender"], r["victim"])]
    k_hat = sum(off.values()) / math.fsum(pair)
    for i, c in enumerate(off.values()):
        assert r["rate_per_pair_exposure"][i] == pytest.approx(c / pair[i], rel=1e-14)
        assert r["ratio_to_null"][i] == pytest.approx(c / pair[i] / k_hat, rel=1e-14)
        assert r["null_rate_per_offender_group"][i] == pytest.approx(k_hat * pop[r["victim"][i]] / N, rel=1e-14)
    with pytest.raises(ValueError, match="offender_on_victim"):
        interracial_rates({"AB": 1}, pop)


def test_probability_of_necessity_and_collider():
    r = probability_of_necessity(0.63, 0.41)
    assert r.necessary_share_bounds["lower"] == pytest.approx(0.22, abs=1e-15)
    assert r.necessary_share_bounds["upper"] == pytest.approx(0.59, abs=1e-15)
    assert r.pn_bounds["lower"] == pytest.approx(0.22 / 0.63, abs=1e-15) and r.pn_monotone == r.pn_bounds["lower"]
    assert probability_of_necessity(0.7, 0.0).identified and not r.identified
    c = collider_arrest(0.31, 0.17, 0.23)
    assert c.population_or == 1 and c.arrestee_or == pytest.approx(0.23, abs=1e-14)
    assert math.fsum(c.cells_among_arrestees.values()) == pytest.approx(1.0, abs=1e-15)
    assert c.arrestee_log_or == pytest.approx(math.log(0.23), abs=1e-13)
