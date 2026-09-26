"""Jackknife replicate weights against R survey::as.svrepdesign.

JK1: svydesign(ids = ~1, weights = ~w) with w = 1:4, type = "JK1".
JKn: svydesign(ids = ~psu, strata = ~st, weights = ~w), st = (1, 1, 2, 2),
one PSU per unit, type = "JKn".
"""

from morie.weights import jackknife_replicate_weights


def test_jk1_matches_survey():
    got = jackknife_replicate_weights([1, 2, 3, 4], [0, 0, 0, 0]).tolist()
    ref = [[0, 4 / 3, 4 / 3, 4 / 3], [8 / 3, 0, 8 / 3, 8 / 3], [4, 4, 0, 4], [16 / 3, 16 / 3, 16 / 3, 0]]
    for g, r in zip(got, ref):
        for a, b in zip(g, r):
            assert abs(a - b) <= 1e-15 * max(1.0, abs(b))


def test_jkn_matches_survey():
    got = jackknife_replicate_weights([1, 2, 3, 4], [1, 1, 2, 2], jk_type="JKn").tolist()
    assert got == [[0.0, 2.0, 1.0, 1.0], [4.0, 0.0, 2.0, 2.0], [3.0, 3.0, 0.0, 6.0], [4.0, 4.0, 8.0, 0.0]]


def test_calibrate_to_totals_raking_matches_survey():
    # survey::calibrate(des, ~ -1 + gm + gf + x, population = c(55, 60, 300), calfun = "raking")
    import math

    from morie import weights as W
    from morie.fn import _frame_core as pd

    n = 30
    g = [("m", "f")[i % 2] for i in range(n)]
    x = [round(1 + abs(math.sin(1.3 * i)) * 3, 3) for i in range(n)]
    w0 = [round(1 + 0.5 * math.cos(0.8 * i) ** 2, 3) for i in range(n)]
    r = W.calibrate_to_totals(w0, pd.DataFrame({"g": g, "x": x}), {"g": {"m": 55, "f": 60}, "x": 300})
    w = r.weights.tolist()
    assert r.converged
    for a, b in zip((w[0], w[1], w[29]), (6.48488579431871, 3.06061865432032, 5.15253922836655)):
        assert abs(a - b) <= 1e-12 * b
    assert abs(sum(a * b for a, b in zip(w, x)) - 300) <= 1e-9
