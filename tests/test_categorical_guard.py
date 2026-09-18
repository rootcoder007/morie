"""Tests for morie.categorical_guard (the Python arm of rmorie module 25)."""
import pytest

from morie.categorical_guard import (
    audit_categories,
    crosstab_verify,
    decode_labelled,
    guard_binary_treatment,
    marginals_verify,
    odds_ratio_check,
    relabel_forensics,
    safe_factor,
    safe_recode,
    safe_relabel,
    transfer_verify,
)


def test_safe_recode_maps_by_name_and_refuses_unmapped():
    r = safe_recode(["W", "B", "O", "W"], {"W": "White", "B": "Black", "O": "Other"})
    assert r["values"] == ["White", "Black", "Other", "White"]
    assert len(r["audit"]["checksum"]) == 64
    with pytest.raises(ValueError, match="values with NO mapping"):
        safe_recode(["W", "B", "X"], {"W": "White", "B": "Black"})
    r2 = safe_recode(["W", "X", None], {"W": "White"}, keep=["X"])
    assert r2["values"] == ["White", "X", None]


def test_safe_factor_declares_levels_and_reference():
    f = safe_factor(["White", "Black", "White"], ["White", "Black"], reference="White")
    assert f["codes"] == [1, 2, 1] and f["reference"] == "White"
    with pytest.raises(ValueError, match="outside the declared levels"):
        safe_factor(["White", "Other"], ["White", "Black"])
    with pytest.raises(ValueError, match="is not levels"):
        safe_factor(["White"], ["White", "Black"], reference="Black")


def test_audit_flags_numeric_codes_and_case_variants():
    rows = audit_categories({"race": ["1", "2", "2", "3"], "city": ["Toronto", "toronto", "Ottawa", "Ottawa"]},
                            factor_levels={"race": ["1", "2", "3"]})
    assert "imported CODES" in rows[0]["hazards"]
    assert "case-variant" in rows[1]["hazards"]


def test_crosstab_verify_catches_a_swap():
    assert crosstab_verify(["W", "B", "W"], ["White", "Black", "White"], {"W": "White", "B": "Black"})
    with pytest.raises(ValueError, match="THIS is how groups get swapped"):
        crosstab_verify(["W", "B"], ["Black", "White"], {"W": "White", "B": "Black"})
    with pytest.raises(ValueError, match="rows were lost"):
        crosstab_verify(["W", "B"], ["White"], {"W": "White"})


def test_marginals_verify_names_the_permutation():
    x = ["White", "White", "Black", "Indigenous", "White", "Black"]
    assert marginals_verify(x, {"White": 3, "Black": 2, "Indigenous": 1})["ok"]
    with pytest.raises(ValueError, match="White -> Black"):
        marginals_verify(x, {"White": 2, "Black": 3, "Indigenous": 1})


def test_odds_ratio_check_recovers_the_label_swap():
    tab = {"A": [900, 100], "B": [700, 300], "C": [400, 600]}
    ok = odds_ratio_check(tab, "A", {"B": 300 / 700 / (100 / 900), "C": 600 / 400 / (100 / 900)})
    assert ok["consistent"] and abs(ok["computed"]["B"] - 3.857142857) < 1e-6
    swapped = odds_ratio_check(tab, "A", {"B": 13.5, "C": 3.857})
    assert not swapped["consistent"] and swapped["matches"][0]["relabelling"] == "B -> C, C -> B"
    assert "mislabelled" in swapped["verdict"]
    nowhere = odds_ratio_check(tab, "A", {"B": 36, "C": 4})
    assert not nowhere["consistent"] and nowhere["matches"] == []


def test_guard_binary_treatment():
    assert guard_binary_treatment([0, 1, 1, 0], "d")
    with pytest.raises(ValueError, match="level INDICES"):
        guard_binary_treatment(["treated", "control"], "d")
    with pytest.raises(ValueError, match="must be binary"):
        guard_binary_treatment([0, 2], "d")


def test_odds_ratio_check_names_the_ohrc_four_way_rotation():
    """White coded as Black, Black as Other, Other as Unknown, Unknown as White
    (OHRC, Correction to "A Disparate Impact", 26 January 2023; Jung 2022)."""
    tab = {"White": [9000, 120], "Black": [2000, 220], "Other": [1500, 60], "Unknown": [4000, 1]}
    ref = 120 / 9000
    correct = odds_ratio_check(tab, "White", {"Black": 220 / 2000 / ref, "Other": 60 / 1500 / ref, "Unknown": 1 / 4000 / ref})
    assert correct["consistent"]
    rot = {"White": tab["Unknown"], "Black": tab["White"], "Other": tab["Black"], "Unknown": tab["Other"]}
    rref = rot["White"][1] / rot["White"][0]
    reported = {k: rot[k][1] / rot[k][0] / rref for k in ("Black", "Other", "Unknown")}
    r = odds_ratio_check(tab, "White", reported)
    assert not r["consistent"] and any("Unknown -> White" in m["relabelling"] for m in r["matches"])
    assert reported["Black"] / correct["computed"]["Black"] > 5


def test_safe_relabel_decode_labelled_and_forensics():
    with pytest.raises(ValueError, match="POSITION"):
        safe_relabel(["W", "B"], ["Black", "White"])
    g = safe_relabel(["W", "B", "O", "W"], {"W": "White", "B": "Black", "O": "Other"})
    assert g["values"] == ["White", "Black", "Other", "White"] and g["levels"] == ["White", "Black", "Other"]
    d = decode_labelled([1, 2, 2, 4], {1: "White", 2: "Black", 3: "Other", 4: "Unknown"})
    assert d["values"] == ["White", "Black", "Black", "Unknown"]
    assert d["levels"] == ["White", "Black", "Other", "Unknown"]
    with pytest.raises(ValueError, match="NO mapping"):
        decode_labelled([1, 5], {1: "White"})
    vl = {1: "White", 2: "Black", 3: "Other", 4: "Unknown"}
    obs = {"White": "Black", "Black": "Other", "Other": "Unknown", "Unknown": "White"}
    r = relabel_forensics(vl, obs)
    assert "labels sorted alphabetically, assigned by code position" in r["matches"]
    assert "reproduced EXACTLY" in r["verdict"]
    assert sorted(vl.values()) == [obs[lab] for lab in vl.values()]
    none = relabel_forensics(vl, {"White": "Other", "Black": "White", "Other": "Black", "Unknown": "Unknown"})
    assert not none["matches"] and "No positional" in none["verdict"]
    a = audit_categories({"race": ["1. White", "2. Black", "1. White"]})
    assert "code prefixes" in a[0]["hazards"]


def test_transfer_verify_spss_to_python():
    codes = [1, 1, 2, 4, 1]
    ok = transfer_verify(codes, {"White": 3, "Black": 1, "Unknown": 1},
                         value_labels={1: "White", 2: "Black", 3: "Other", 4: "Unknown"},
                         code_book={1: "White", 2: "Black", 3: "Other", 4: "Unknown"})
    assert ok["ok"] and ok["code_book_ok"]
    assert ok["decoded"]["levels"] == ["White", "Black", "Other", "Unknown"]
    with pytest.raises(ValueError, match="disagree with the source code book"):
        transfer_verify(codes, {"White": 3, "Black": 1, "Unknown": 1},
                        value_labels={1: "Black", 2: "Other", 3: "Unknown", 4: "White"},
                        code_book={1: "White", 2: "Black", 3: "Other", 4: "Unknown"})
    with pytest.raises(ValueError):
        transfer_verify(["Black", "Black", "Other", "White", "Black"], {"White": 3, "Black": 1, "Unknown": 1})


def test_round_3_strict_false_forensics_identity_and_whitespace_hazards():
    x = ["White", "White", "Black", "Indigenous", "White", "Black"]
    r = marginals_verify(x, {"White": 2, "Black": 3, "Indigenous": 1}, strict=False)
    assert r["ok"] is False and r["permutation"]["White"] == "Black" and "permuted" in r["message"]
    t = transfer_verify(["Black", "White", "White", "Black", "Black"], {"White": 3, "Black": 2}, strict=False)
    assert t["ok"] is False and t["marginals"]["permutation"]["White"] == "Black" and t["reasons"]
    ident = relabel_forensics({1: "White", 2: "Black"}, {"White": "White", "Black": "Black"})
    assert ident["matches"] == [] and "no permutation" in ident["verdict"]
    a = audit_categories({"trailing": ["White ", "White", "Black", "Black"],
                          "leading": [" White", "White", "Black", "Black"],
                          "na_string": ["White", "NA", "Black", "Black"],
                          "empty_str": ["", "White", "Black", "Black"],
                          "unicode_ws": ["White\u00a0", "White", "Black", "Black"]})
    h = {row["column"]: row["hazards"] for row in a}
    assert "trailing whitespace" in h["trailing"] and "REFERENCE level" in h["leading"]
    assert "sentinel" in h["na_string"] and "empty-string" in h["empty_str"]
    assert "non-breaking" in h["unicode_ws"]
