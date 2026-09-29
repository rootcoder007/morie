"""Tests for morie.fn.sntmn: values recomputed from the definition."""

from morie.fn.sntmn import sentence_mandatory_min


def test_shares_and_excess():
    d = {
        "offense": ["a", "a", "b", "b", "b"],
        "sentence_days": [60.0, 90.0, 30.0, 40.0, 100.0],
        "mandatory_min_days": [60.0, 60.0, 30.0, 60.0, 60.0],
    }
    r = sentence_mandatory_min(d)
    assert r.value == 2 / 5
    assert r.extra["pct_below_minimum"] == 1 / 5
    assert r.extra["pct_above_minimum"] == 2 / 5
    assert r.extra["mean_above_minimum"] == (30.0 + 40.0) / 2
    assert r.extra["by_offense"]["b"]["pct_at_minimum"] == 1 / 3
