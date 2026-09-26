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
