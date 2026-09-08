"""Tests for icc12c.icc_two_way.

Anchored on the worked example of Shrout and Fleiss (1979), Psychological
Bulletin 86(2):420-428: Table 2 (p. 423) is the data, Table 3 (p. 423) the
ANOVA, Table 4 (p. 424) the six published coefficients.
"""

import pytest

from morie.fn.icc12c import icc_two_way

# Table 2, p. 423: four ratings on six targets
SF = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
      [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]


def test_shrout_fleiss_table_3_mean_squares():
    r = icc_two_way(SF)
    assert round(r["BMS"], 2) == 11.24
    assert round(r["WMS"], 2) == 6.26
    assert round(r["JMS"], 2) == 32.49
    assert round(r["EMS"], 2) == 1.02


def test_shrout_fleiss_table_4_coefficients():
    r = icc_two_way(SF)
    assert round(r["icc11"], 2) == 0.17
    assert round(r["icc21"], 2) == 0.29
    assert round(r["icc31"], 2) == 0.71
    assert round(r["icc1k"], 2) == 0.44
    assert round(r["icc2k"], 2) == 0.62
    assert round(r["icc3k"], 2) == 0.91


def test_icc3k_is_cronbach_alpha():
    """ICC(3,k) is Cronbach's alpha (p. 426); computed here directly."""
    r = icc_two_way(SF)
    k = 4
    col_var = []
    for j in range(k):
        col = [row[j] for row in SF]
        m = sum(col) / len(col)
        col_var.append(sum((v - m) ** 2 for v in col) / (len(col) - 1))
    tot = [sum(row) for row in SF]
    mt = sum(tot) / len(tot)
    vt = sum((v - mt) ** 2 for v in tot) / (len(tot) - 1)
    alpha = k / (k - 1) * (1 - sum(col_var) / vt)
    assert abs(alpha - r["icc3k"]) < 1e-12


def test_model_selector_covers_all_six_forms():
    r = icc_two_way(SF)
    for lab, key in (("ICC(1,1)", "icc11"), ("1-k", "icc1k"), ("2,1", "icc21"),
                     ("icc(2,k)", "icc2k"), ("3 1", "icc31"), ("3k", "icc3k")):
        assert icc_two_way(SF, lab)["estimate"] == r[key]
    # the paper labels its own table with the concrete k
    assert icc_two_way(SF, "ICC(1,4)")["estimate"] == r["icc1k"]


def test_constant_judge_offsets_are_consistency_not_agreement():
    base = [3.0, 5.0, 1.0, 8.0, 2.0, 6.0]
    X = [[b + o for o in (0.0, 2.0, -1.0, 0.5)] for b in base]
    r = icc_two_way(X)
    assert abs(r["EMS"]) < 1e-12
    assert abs(r["icc31"] - 1.0) < 1e-12
    assert r["icc21"] < 1.0 - 1e-6
    assert r["icc11"] < r["icc21"] < r["icc31"]


def test_error_paths():
    with pytest.raises(ValueError):
        icc_two_way([[1.0, 2.0]])
    with pytest.raises(ValueError):
        icc_two_way([[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError):
        icc_two_way(SF, "ICC(4,2)")
    with pytest.raises(ValueError):
        icc_two_way(SF, "")
