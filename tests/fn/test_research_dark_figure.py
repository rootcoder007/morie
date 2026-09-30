import math

import pytest

from morie.fn.research_dark_figure import (
    dark_figure_bounds,
    dark_figure_breakdown,
    dark_figure_hierarchy,
    dark_figure_three_list,
    dark_figure_two_source,
)


def test_two_source_petersen_box_and_chapman():
    r = dark_figure_two_source(437, 251.5, 73, kappa=1.7, chapman=True)
    assert r.point == pytest.approx(437 * 251.5 / 73, rel=1e-15)
    assert r.lower == pytest.approx(r.point / 1.7, rel=1e-15) and r.upper == pytest.approx(1.7 * r.point, rel=1e-15)
    assert r.chapman == pytest.approx(438 * 252.5 / 74 - 1, rel=1e-15)
    assert dark_figure_two_source(400, 250, 80).theorem == "Research.P1.TwoSource.lincoln_petersen"
    with pytest.raises(ValueError, match="smaller list"):
        dark_figure_two_source(10, 5, 6)


def test_true_rate_bounds_contain_every_admissible_noise_pair():
    a, b = 0.012, 0.37
    v_obs, r = [0.06, 0.013, 0.2, 0.0], [0.02, 0.015, 0.05, 0.0]
    d = dark_figure_bounds(v_obs, r, a, b)
    for k in range(4):
        lo = max(r[k], (v_obs[k] - a) / (1 - a))
        up = max(min(1.0, v_obs[k] / (1 - b)), lo)
        assert d["v_lower"][k] == pytest.approx(lo, abs=1e-15) and d["v_upper"][k] == pytest.approx(up, abs=1e-15)
        assert d["dark_upper"][k] == pytest.approx(up - r[k], abs=1e-15)
        # every noise pair on a grid of the box gives a true rate inside the interval
        for al in (0.0, a / 2, a):
            for be in (0.0, b / 2, b):
                v = (v_obs[k] - al) / (1 - al - be)
                if r[k] <= v <= 1:
                    assert d["v_lower"][k] - 1e-12 <= v <= d["v_upper"][k] + 1e-12
    assert math.isnan(d["ratio_upper"][3])
    with pytest.raises(ValueError, match="below 1"):
        dark_figure_bounds(0.1, 0.05, 0.5, 0.5)


def test_breakdown_is_the_survey_ceiling_crossing():
    b = dark_figure_breakdown(0.073, 0.11)
    assert b.breakdown_beta_max == pytest.approx(1 - 0.073 / 0.11, abs=1e-15)
    assert 0.073 / (1 - b.breakdown_beta_max) == pytest.approx(0.11, abs=1e-12)
    with pytest.raises(ValueError, match="above v_obs"):
        dark_figure_breakdown(0.2, 0.1)


def test_three_list_closed_form_pairs_and_floors():
    c = {"100": 131, "010": 87.5, "001": 64, "110": 41, "101": 29, "011": 23, "111": 11}
    t = dark_figure_three_list(c, candidate_missing=[55.5, 300])
    m000 = 11 * 131 * 87.5 * 64 / (41 * 29 * 23)
    assert t.observed == math.fsum(c.values())
    assert t.missing_no_three_way == pytest.approx(m000, rel=1e-14)
    n1 = 131 + 41 + 29 + 11
    n2 = 87.5 + 41 + 23 + 11
    m12 = 41 + 11
    p = t.pairwise[0]
    assert (p["n1"], p["n2"], p["m"]) == (n1, n2, m12)
    assert p["petersen"] == pytest.approx(n1 * n2 / m12, rel=1e-15)
    for q in t.pairwise:
        assert q["petersen"] >= q["floor"] and q["chapman"] >= q["floor"]
    lg = {k: math.log(v) for k, v in c.items()}
    for mm, tw in zip([55.5, 300], t.implied_three_way["three_way"]):
        want = lg["111"] - lg["110"] - lg["101"] - lg["011"] + lg["100"] + lg["010"] + lg["001"] - math.log(mm)
        assert tw == pytest.approx(want, abs=1e-12)
    # the no-three-way missing cell implies a zero three-way interaction
    t0 = dark_figure_three_list(c, candidate_missing=[m000])
    assert t0.implied_three_way["three_way"][0] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="seven cells"):
        dark_figure_three_list({"100": 1})


def test_hierarchy_bounds():
    k = [1, 3, 2, 1, 1, 5, 2, 1, 1]
    h = dark_figure_hierarchy(k)
    assert h.offences == sum(k) and h.incidents == len(k)
    assert h.mean_extra == pytest.approx(sum(v - 1 for v in k) / len(k), abs=1e-15)
    assert h.bounds == {"lower": 9, "upper": 45}
    assert h.bounds["lower"] <= h.offences <= h.bounds["upper"]
    with pytest.raises(ValueError, match="at least one offence"):
        dark_figure_hierarchy([1, 0])
