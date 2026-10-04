"""The native engines shared with rmorie: matching (MatchIt's rules), survey designs (Taylor
linearisation), the g-formula's sandwich SE.

The expected values on the deterministic data below were computed by the R arm (rmorie 1.4.0,
itself cross-validated against MatchIt, optmatch, survey and stdReg), so each test is a
cross-language parity check; the brute-force tests need neither arm.
"""

import itertools
import math

import pytest

from morie import matching as M
from morie import survey as S
from morie._matchit_native import full_match, mm_subclass, mm_weights, nn_match, sap_rect, subclass_scoot
from morie.effects import estimate_ate_gcomputation
from morie.fn import _frame_core as pd


def _data():
    i = range(1, 61)
    d = [1 if (k * 7) % 5 < 2 else 0 for k in i]
    return pd.DataFrame(
        {
            "x1": [math.sin(k) for k in i],
            "x2": [math.cos(1.3 * k) for k in i],
            "d": d,
            "y": [0.5 * dk + math.sin(k) + 0.3 * math.cos(2.1 * k) for dk, k in zip(d, i)],
            "w": [1 + (k % 4) / 2 for k in i],
            "s": [["a", "b", "c"][k % 3] for k in i],
            "psu": [(k - 1) // 3 + 1 for k in i],
        }
    )


def _pairs(m):
    return sorted(f"{a} {b}" for a, b in zip(m.match_pairs["treated_idx"], m.match_pairs["control_idx"]))


NN2 = [
    "12 36",
    "14 13",
    "14 40",
    "17 51",
    "19 26",
    "19 53",
    "2 35",
    "22 5",
    "24 3",
    "24 6",
    "27 0",
    "29 50",
    "29 56",
    "32 45",
    "32 58",
    "34 10",
    "37 1",
    "37 8",
    "39 41",
    "4 23",
    "4 38",
    "42 16",
    "42 18",
    "44 15",
    "47 28",
    "47 46",
    "49 20",
    "52 11",
    "52 43",
    "54 55",
    "57 21",
    "57 48",
    "59 31",
    "7 30",
    "9 25",
    "9 33",
]
VR = [
    "12 21",
    "12 36",
    "14 13",
    "17 16",
    "17 51",
    "19 53",
    "2 35",
    "2 50",
    "22 38",
    "22 5",
    "24 3",
    "27 0",
    "27 46",
    "29 56",
    "32 58",
    "34 10",
    "34 25",
    "37 8",
    "39 11",
    "39 41",
    "4 23",
    "42 18",
    "44 15",
    "44 40",
    "47 28",
    "49 1",
    "49 20",
    "52 43",
    "54 26",
    "54 55",
    "57 48",
    "59 31",
    "59 45",
    "7 30",
    "7 6",
    "9 33",
]
VRW = [
    0.75,
    0.75,
    1,
    1.5,
    1,
    0.75,
    0.75,
    1,
    1.5,
    1,
    0.75,
    0.75,
    1,
    1.5,
    1,
    0.75,
    0.75,
    1,
    1.5,
    1,
    0.75,
    0.75,
    1,
    1.5,
    1,
    0.75,
    0.75,
    1,
    1.5,
    1,
    0.75,
    0.75,
    1,
    1.5,
    1,
    0.75,
    0.75,
    1,
    0.75,
    1,
    0.75,
    0.75,
    1,
    1.5,
    1,
    0.75,
    0.75,
    1,
    1.5,
    1,
    0.75,
    0.75,
    1,
    1.5,
    1,
    0.75,
    1.5,
    1,
    1.5,
    1,
]
FULL_SUB = [
    1,
    2,
    3,
    4,
    5,
    1,
    2,
    6,
    4,
    7,
    8,
    2,
    9,
    4,
    10,
    11,
    2,
    8,
    7,
    12,
    6,
    2,
    13,
    5,
    12,
    2,
    2,
    14,
    15,
    12,
    2,
    2,
    14,
    16,
    1,
    2,
    9,
    4,
    10,
    11,
    2,
    11,
    16,
    17,
    9,
    2,
    11,
    15,
    17,
    9,
    2,
    8,
    16,
    12,
    6,
    3,
    14,
    17,
    13,
    2,
]
SUB4 = [
    2,
    1,
    1,
    3,
    4,
    2,
    1,
    1,
    3,
    4,
    2,
    1,
    2,
    3,
    3,
    2,
    1,
    2,
    4,
    3,
    1,
    1,
    2,
    4,
    3,
    1,
    1,
    2,
    4,
    3,
    1,
    1,
    3,
    4,
    2,
    1,
    1,
    3,
    4,
    2,
    1,
    2,
    4,
    4,
    1,
    1,
    2,
    4,
    4,
    1,
    1,
    2,
    4,
    3,
    1,
    1,
    2,
    4,
    2,
    1,
]


def test_matchers_reproduce_the_r_arm():
    df = _data()
    with pytest.warns(UserWarning, match="Fewer control units"):
        assert _pairs(M.match_nearest_neighbor(df, "d", ["x1", "x2"], n_neighbors=2)) == NN2
    vr = M.match_variable_ratio(df, "d", ["x1", "x2"], min_ratio=1, max_ratio=3, caliper=None)
    assert _pairs(vr) == VR
    assert vr.matched_data["weights"].tolist() == pytest.approx(VRW, abs=1e-12)
    fm = M.match_full(df, "d", ["x1", "x2"])
    assert fm.details["total_distance"] == pytest.approx(0.480339044381522, abs=1e-8)
    assert fm.matched_data["subclass"].tolist() == FULL_SUB
    sc, eff = M.subclassify(df, "d", ["x1", "x2"], n_strata=4)
    assert sc["subclass"].tolist() == SUB4
    assert sum(eff["n_treated"]) + sum(eff["n_control"]) == len(df)


def test_survey_design_reproduces_the_r_arm():
    d = S.SurveyDesign(_data(), "w", strata_col="s", cluster_col="psu", nest=True)
    m = d.mean("y")
    assert [m["mean"], m["se"]] == pytest.approx([0.235771971916831, 0.0985475044564735], abs=1e-12)
    g = d.svyglm("y ~ x1", family="gaussian")
    assert [*g.params, *g.bse] == pytest.approx(
        [0.201486345724238, 0.965871608933961, 0.0347789223480888, 0.0612278844444572], abs=1e-12
    )


def test_g_formula_sandwich_reproduces_the_r_arm():
    r = estimate_ate_gcomputation(_data(), treatment="d", outcome="y", covariates=["x1", "x2"])
    assert [r["ate"], r["se"]] == pytest.approx([0.499637935705705, 0.0588985530317521], abs=1e-12)
    assert r["ci_lower"] == pytest.approx(r["ate"] - 1.959963984540054 * r["se"])


def test_full_match_is_a_minimum_edge_cover_by_brute_force():
    for seed in range(6):
        nt, nc = 2 + seed % 2, 2 + seed % 3
        p = [0.5 + 0.4 * math.sin(3.7 * (k + 1) + seed) for k in range(nt + nc)]
        tr = [1] * nt + [0] * nc
        sub, edges = full_match(p, tr)
        got = sum(abs(p[t] - p[c]) for t, c in edges)
        E = list(itertools.product(range(nt), range(nt, nt + nc)))
        best = math.inf
        for mask in range(1, 2 ** len(E)):
            use = [E[k] for k in range(len(E)) if mask >> k & 1]
            if {t for t, _ in use} == set(range(nt)) and {c for _, c in use} == set(range(nt, nt + nc)):
                best = min(best, sum(abs(p[t] - p[c]) for t, c in use))
        assert got == pytest.approx(best, abs=1e-12)
        for s in set(sub):
            members = [k for k in range(nt + nc) if sub[k] == s]
            assert sum(tr[k] for k in members) == 1 or sum(1 - tr[k] for k in members) == 1


def test_matcher_pieces_follow_matchits_rules():
    # rounds: the higher-scoring treated unit chooses first; a caliper bars far controls
    tr = [1, 1, 0, 0, 0, 0]
    d = [0.50, 0.40, 0.48, 0.47, 0.10, 0.90]
    rows = nn_match(tr, d, [2, 2])
    assert rows[0][0] == 2 and rows[1][0] == 3
    assert sorted(c for r in nn_match(tr, d, [2, 2], caliper=0.15) for c in r) == [2, 3]
    assert nn_match(tr, d, [1, 1], replace=True) == [[2], [3]]
    with pytest.raises(ValueError, match="0/1"):
        nn_match([1, 2], [0.0, 1.0], [1])
    with pytest.raises(ValueError, match="finite"):
        nn_match([1, 0], [float("nan"), 1.0], [1])
    with pytest.raises(ValueError, match="one entry"):
        nn_match([1, 0], [0.0, 1.0], [1, 1])
    assert sap_rect([[4, 2, 3], [1, 0, 2], [3, 5, 2]]) == [1, 0, 2]
    with pytest.raises(ValueError, match="ncol >= nrow"):
        sap_rect([[1.0], [2.0]])
    sub = subclass_scoot([1, 1, 2, 2, 3, 3, 3], [1, 0, 1, 1, 1, 0, 0], [0.1, 0.2, 0.4, 0.5, 0.7, 0.75, 0.8])
    assert sub[5] == 2
    with pytest.raises(ValueError, match="not enough units"):
        subclass_scoot([1, 2, 2], [1, 1, 0], [1.0, 2.0, 3.0])
    rows = [[2, 4], [3], []]
    assert mm_weights(rows, 6, [0, 1, 5], [1, 1, 0, 0, 0, 1]) == pytest.approx([1, 1, 0.75, 1.5, 0.75, 0])
    assert mm_subclass(rows, 6, [0, 1, 5]) == [1, 2, 1, 2, 1, None]


def test_survey_design_checks_and_fpc():
    df = _data()
    with pytest.raises(ValueError, match="not nested in strata"):
        S.SurveyDesign(df, "w", strata_col="s", cluster_col="psu")
    with pytest.raises(ValueError, match="not in data"):
        S.SurveyDesign(df, "nope")
    one = df.iloc[[0, 1, 2, 3]].copy()
    one["s"] = ["a", "a", "b", "b"]
    one["psu"] = [1, 1, 2, 3]
    with pytest.raises(ValueError, match="only one PSU"):
        S.SurveyDesign(one, "w", strata_col="s", cluster_col="psu", nest=True).mean("y")
    # a census stratum adds nothing; an fpc of sampling fractions is read as N = n / f
    two = df.iloc[list(range(12))].copy()
    two["s"] = ["a"] * 6 + ["b"] * 6
    two["N"] = [6.0] * 6 + [600.0] * 6
    d = S.SurveyDesign(two, "w", strata_col="s", fpc_col="N")
    ys = [float(v) for v in two["y"].tolist()]
    ws = [float(v) for v in two["w"].tolist()]
    m = sum(a * b for a, b in zip(ws, ys)) / sum(ws)
    zb = [ws[k] * (ys[k] - m) / sum(ws) for k in range(6, 12)]
    mz = sum(zb) / 6
    want = math.sqrt((600 - 6) / 600 * 6 / 5 * sum((z - mz) ** 2 for z in zb))
    assert d.mean("y")["se"] == pytest.approx(want, rel=1e-12)
    with pytest.raises(ValueError, match="positive"):
        S.SurveyDesign(two.assign(N=[-1.0] * 12), "w", strata_col="s", fpc_col="N")
    assert math.isnan(S.SurveyDesign(df.assign(y=[float("nan")] + df["y"].tolist()[1:]), "w").mean("y")["mean"])


def test_matchers_refuse_bad_arguments():
    df = _data()
    all_t = df.assign(d=[1] * len(df))
    for f in (M.match_full, M.subclassify, M.match_variable_ratio):
        with pytest.raises(ValueError, match="both treated and control"):
            f(all_t, "d", ["x1"])
    with pytest.raises(ValueError, match="n_strata"):
        M.subclassify(df, "d", ["x1"], n_strata=0)
    with pytest.raises(ValueError, match="must not exceed"):
        M.match_variable_ratio(df, "d", ["x1"], min_ratio=3, max_ratio=2)
    with pytest.raises(ValueError, match="caliper"):
        M.match_variable_ratio(df, "d", ["x1"], caliper=-1)
